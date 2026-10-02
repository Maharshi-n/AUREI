import argparse
import asyncio
import json
import queue
import threading
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from engine import Engine, Options, Text, sample


class CapacityError(Exception):
    pass


@dataclass
class Job:
    ids: list
    options: Options
    stop: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    output: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=32))
    cancelled: threading.Event = field(default_factory=threading.Event)
    slot: int = -1
    tokens: list = field(default_factory=list)
    decoder: object = None
    rng: object = None


class Scheduler:
    def __init__(self, engine, active=2, waiting=8):
        if not 1 <= active <= 2 or waiting < 1:
            raise ValueError("Use 1 or 2 active requests and a positive queue limit")
        self.engine = engine
        self.limit = active
        self.waiting = waiting
        self.cache = engine.model.cache(active)
        self.pending = deque()
        self.jobs = {}
        self.active = []
        self.free = list(range(active))
        self.lock = threading.Condition()
        self.closed = False
        self.error = None
        self.batch_sizes = {}
        self.thread = threading.Thread(target=self.run, name="inference", daemon=True)
        self.thread.start()

    def submit(self, ids, options, stop=True):
        options.validate(len(ids), self.engine.context)
        with self.lock:
            if self.closed or self.error:
                raise CapacityError(self.error or "Engine is shutting down")
            if len(self.pending) >= self.waiting:
                raise CapacityError("Request queue is full; retry later")
            job = Job(list(ids), options, stop)
            self.jobs[job.id] = job
            self.pending.append(job)
            self.lock.notify()
            return job

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            job.cancelled.set()
            self.send(job, {"event": "error", "id": job.id, "error": "Request cancelled"})
            if job in self.pending:
                self.pending.remove(job)
                self.jobs.pop(job.id, None)
            self.lock.notify()
            return True

    def status(self):
        with self.lock:
            return {"ready": not self.closed and self.error is None,
                    "active": len(self.active), "queued": len(self.pending),
                    "max_active": self.limit, "max_queue": self.waiting,
                    "context": self.engine.context, "device": str(self.engine.device),
                    "dtype": str(self.engine.dtype), "cache_bytes": self.cache.bytes,
                    "decode_batches": dict(self.batch_sizes), "error": self.error}

    def send(self, job, event):
        try:
            job.output.put_nowait(event)
            return True
        except queue.Full:
            job.cancelled.set()
            while not job.output.empty():
                try:
                    job.output.get_nowait()
                except queue.Empty:
                    break
            job.output.put_nowait({"event": "error", "id": job.id, "error": "Client consumed output too slowly"})
            return False

    def release(self, job):
        if job.slot >= 0:
            self.cache.clear(job.slot)
            self.free.append(job.slot)
            job.slot = -1
        with self.lock:
            if job in self.active:
                self.active.remove(job)
            self.jobs.pop(job.id, None)

    def token(self, job, logits):
        if job.cancelled.is_set():
            self.release(job)
            return
        token = sample(logits, job.options, job.rng)
        job.tokens.append(token)
        reason = "stop" if job.stop and token in self.engine.stop else "length" if len(job.tokens) >= job.options.max_tokens else None
        text = job.decoder.push(token)
        if reason:
            text += job.decoder.finish()
        ok = self.send(job, {"event": "token", "id": job.id, "token": token, "text": text})
        if reason and ok:
            self.send(job, {"event": "done", "id": job.id, "finish_reason": reason,
                            "usage": {"prompt_tokens": len(job.ids), "completion_tokens": len(job.tokens)}})
        if reason or not ok:
            self.release(job)

    @torch.inference_mode()
    def run(self):
        try:
            while True:
                with self.lock:
                    while not self.closed and not self.active and not self.pending:
                        self.lock.wait()
                    if self.closed:
                        break
                for job in list(self.active):
                    if job.cancelled.is_set():
                        self.release(job)
                with self.lock:
                    job = self.pending.popleft() if self.pending and self.free else None
                if job is not None:
                    if job.cancelled.is_set():
                        self.release(job)
                        continue
                    job.slot = self.free.pop(0)
                    job.decoder = Text(self.engine.tokenizer)
                    job.rng = torch.Generator(device=self.engine.device).manual_seed(job.options.seed)
                    with self.lock:
                        self.active.append(job)
                    ids = torch.tensor([job.ids], device=self.engine.device)
                    logits = self.engine.model.forward(ids, self.cache, [job.slot])
                    self.token(job, logits[0, -1])
                batch = list(self.active)
                if batch:
                    ids = torch.tensor([[j.tokens[-1]] for j in batch], device=self.engine.device)
                    logits = self.engine.model.forward(ids, self.cache, [j.slot for j in batch])
                    with self.lock:
                        self.batch_sizes[len(batch)] = self.batch_sizes.get(len(batch), 0) + 1
                    for i, job in enumerate(batch):
                        self.token(job, logits[i, -1])
        except Exception as error:
            self.error = "GPU memory exhausted; lower context or active limit and restart" if isinstance(error, torch.OutOfMemoryError) else f"Inference worker failed: {type(error).__name__}: {error}"
        finally:
            with self.lock:
                self.closed = True
                jobs = list(self.jobs.values())
                self.pending.clear()
            for job in jobs:
                self.send(job, {"event": "error", "id": job.id, "error": self.error or "Engine shut down"})
                self.release(job)

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.thread.join()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=32768)
    max_tokens: int = Field(default=128, ge=1, le=1024, strict=True)
    temperature: float = Field(default=0, ge=0, le=5, allow_inf_nan=False)
    top_p: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, lt=2**63, strict=True)


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in {"/generate", "/stream"}:
            return await self.app(scope, receive, send)
        total, parts = 0, []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            total += len(part)
            if total > 131072:
                response = JSONResponse({"detail": "Request body exceeds 128 KiB"}, status_code=413)
                return await response(scope, receive, send)
            parts.append(part)
            if not message.get("more_body", False):
                break
        body = b"".join(parts)
        used = False
        async def replay():
            nonlocal used
            if not used:
                used = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


async def events(request, scheduler, job, disconnect=True):
    try:
        while True:
            if disconnect and await request.is_disconnected():
                break
            try:
                event = await asyncio.to_thread(job.output.get, True, 0.1)
            except queue.Empty:
                continue
            yield event
            if event["event"] in {"done", "error"}:
                break
    finally:
        scheduler.cancel(job.id)


def create_app(device="auto", dtype=None, context=1024, active=2, waiting=8, engine=None):
    @asynccontextmanager
    async def lifespan(app):
        e = engine or await asyncio.to_thread(Engine, device, dtype, context)
        try:
            app.state.scheduler = await asyncio.to_thread(Scheduler, e, active, waiting)
        except torch.OutOfMemoryError as error:
            raise RuntimeError("Not enough memory for the cache; use --active 1 or a smaller --context") from error
        yield
        await asyncio.to_thread(app.state.scheduler.close)

    app = FastAPI(title="AUREI", version="1.0", lifespan=lifespan)

    app.add_middleware(BodyLimit)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, error):
        details = [{"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in error.errors()]
        return JSONResponse({"detail": details}, status_code=422)

    def submit(request, data):
        s = request.app.state.scheduler
        try:
            ids = s.engine.prompt(data.prompt)
            return s, s.submit(ids, Options(data.max_tokens, data.temperature, data.top_p, data.seed))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except CapacityError as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from error

    @app.get("/health")
    async def health(request: Request):
        status = request.app.state.scheduler.status()
        return JSONResponse(status, status_code=200 if status["ready"] else 503)

    @app.post("/generate")
    async def generate(data: Input, request: Request):
        s, job = submit(request, data)
        text, tokens = [], []
        async for event in events(request, s, job):
            if event["event"] == "error":
                raise HTTPException(503, event["error"])
            if event["event"] == "token":
                text.append(event["text"])
                tokens.append(event["token"])
            if event["event"] == "done":
                return {"id": job.id, "text": "".join(text), "tokens": tokens,
                        "finish_reason": event["finish_reason"], "usage": event["usage"]}
        raise HTTPException(499, "Request cancelled")

    @app.post("/stream")
    async def stream(data: Input, request: Request):
        s, job = submit(request, data)

        async def body():
            try:
                yield f"event: start\ndata: {json.dumps({'id': job.id})}\n\n"
                async for event in events(request, s, job, disconnect=False):
                    yield f"event: {event['event']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                s.cancel(job.id)

        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.delete("/requests/{job_id}")
    async def cancel(job_id: str, request: Request):
        if not request.app.state.scheduler.cancel(job_id):
            raise HTTPException(404, "Request is no longer active")
        return {"id": job_id, "cancelled": True}

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Serve the local Qwen engine")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--active", type=int, default=2)
    p.add_argument("--queue", type=int, default=8)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(create_app(args.device, args.dtype, args.context, args.active, args.queue),
                host="127.0.0.1", port=args.port, workers=1, timeout_graceful_shutdown=5)
