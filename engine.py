import codecs
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache" / "huggingface" / "hub")
os.environ["HF_XET_CACHE"] = str(ROOT / ".cache" / "huggingface" / "xet")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_DIR = ROOT / "models" / "qwen3-0.6b"


def download():
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=MODEL_DIR,
        cache_dir=ROOT / ".cache" / "huggingface" / "hub",
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "vocab.json", "merges.txt"],
        max_workers=2,
    )
    (MODEL_DIR / "revision.txt").write_text(REVISION, encoding="utf-8")
    return MODEL_DIR


def device_info(device="auto", dtype=None):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install a CUDA PyTorch build or use --device cpu")
    dtype = dtype or ("float16" if device.type == "cuda" else "float32")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Use float32, float16 or bfloat16")
    if device.type == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device, getattr(torch, dtype)


def norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * weight


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


class Cache:
    def __init__(self, model, slots):
        if slots < 1:
            raise ValueError("Cache needs at least one slot")
        c = model.config
        shape = (c["num_hidden_layers"], slots, c["num_key_value_heads"], model.context, c["head_dim"])
        self.k = torch.zeros(shape, device=model.device, dtype=model.dtype)
        self.v = torch.zeros_like(self.k)
        self.lengths = [0] * slots

    def clear(self, slot):
        self.k[:, slot].zero_()
        self.v[:, slot].zero_()
        self.lengths[slot] = 0

    @property
    def bytes(self):
        return 2 * self.k.numel() * self.k.element_size()


class Model:
    def __init__(self, config, weights, device, dtype, context=1024):
        if not 1 <= context <= config["max_position_embeddings"]:
            raise ValueError("Invalid total context limit")
        self.config = config
        self.w = weights
        self.device = torch.device(device)
        self.dtype = dtype
        self.context = context
        d = config["head_dim"]
        freq = 1.0 / (config["rope_theta"] ** (torch.arange(0, d, 2, device=self.device).float() / d))
        angle = torch.arange(context, device=self.device).float()[:, None] * freq
        angle = torch.cat((angle, angle), dim=-1)
        self.cos = angle.cos().to(dtype)
        self.sin = angle.sin().to(dtype)

    def cache(self, slots=1):
        return Cache(self, slots)

    @torch.inference_mode()
    def forward(self, ids, cache=None, slots=None, sizes=None, all_logits=False):
        if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError("ids must be a nonempty [batch, tokens] tensor")
        batch, count = ids.shape
        sizes = sizes if sizes is not None else [count] * batch
        if len(sizes) != batch or any(n < 1 or n > count for n in sizes):
            raise ValueError("Invalid sequence lengths")
        slots = slots if slots is not None else list(range(batch))
        if len(slots) != batch or len(set(slots)) != batch:
            raise ValueError("Each row needs a different cache slot")
        if cache is not None and any(s < 0 or s >= len(cache.lengths) for s in slots):
            raise ValueError("Invalid cache slot")
        starts = [cache.lengths[s] for s in slots] if cache is not None else [0] * batch
        ends = [a + b for a, b in zip(starts, sizes)]
        if max(ends) > self.context:
            raise ValueError("Input exceeds total context limit")
        c, w = self.config, self.w
        pos = torch.tensor(starts, device=self.device)[:, None] + torch.arange(count, device=self.device)
        pos = pos.clamp_max(self.context - 1)
        cos, sin = self.cos[pos][:, None], self.sin[pos][:, None]
        keys = torch.arange(max(ends), device=self.device)
        mask = (keys[None, None, None, :] <= pos[:, None, :, None])
        mask = mask & (keys[None, None, None, :] < torch.tensor(ends, device=self.device)[:, None, None, None])
        causal = False
        if all(n == count for n in sizes) and len(set(ends)) == 1 and (count == 1 or max(starts) == 0):
            mask = None
            causal = count > 1
        x = F.embedding(ids, w["model.embed_tokens.weight"])
        heads, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        eps = c["rms_norm_eps"]
        for i in range(c["num_hidden_layers"]):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            y = norm(x, w[p + "input_layernorm.weight"], eps)
            q = F.linear(y, w[a + "q_proj.weight"]).view(batch, count, heads, d)
            k = F.linear(y, w[a + "k_proj.weight"]).view(batch, count, kv, d)
            v = F.linear(y, w[a + "v_proj.weight"]).view(batch, count, kv, d).transpose(1, 2)
            q = rotate(norm(q, w[a + "q_norm.weight"], eps).transpose(1, 2), cos, sin)
            k = rotate(norm(k, w[a + "k_norm.weight"], eps).transpose(1, 2), cos, sin)
            if cache is not None:
                for row, slot in enumerate(slots):
                    cache.k[i, slot, :, starts[row]:ends[row]] = k[row, :, :sizes[row]]
                    cache.v[i, slot, :, starts[row]:ends[row]] = v[row, :, :sizes[row]]
                k = cache.k[i, slots, :, :max(ends)]
                v = cache.v[i, slots, :, :max(ends)]
            else:
                k, v = k[:, :, :max(ends)], v[:, :, :max(ends)]
            gqa = self.device.type == "cuda" and mask is None
            if not gqa:
                k = k.repeat_interleave(heads // kv, dim=1)
                v = v.repeat_interleave(heads // kv, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0,
                                             is_causal=causal, enable_gqa=gqa)
            y = y.transpose(1, 2).reshape(batch, count, heads * d)
            x = x + F.linear(y, w[a + "o_proj.weight"])
            y = norm(x, w[p + "post_attention_layernorm.weight"], eps)
            m = p + "mlp."
            y = F.silu(F.linear(y, w[m + "gate_proj.weight"])) * F.linear(y, w[m + "up_proj.weight"])
            x = x + F.linear(y, w[m + "down_proj.weight"])
        if not all_logits:
            x = x[torch.arange(batch, device=self.device), torch.tensor(sizes, device=self.device) - 1][:, None]
        x = norm(x, w["model.norm.weight"], eps)
        logits = F.linear(x, w["model.embed_tokens.weight"])
        if cache is not None:
            for slot, end in zip(slots, ends):
                cache.lengths[slot] = end
        return logits


@dataclass
class Options:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.95
    seed: int = 42

    def validate(self, prompt_length, context):
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 5:
            raise ValueError("temperature must be between 0 and 5")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        if prompt_length < 1 or prompt_length + self.max_tokens > context:
            raise ValueError(f"Prompt plus max_tokens must fit within {context} tokens")


def sample(logits, options, rng):
    if options.temperature == 0:
        return int(logits.argmax().item())
    probs = torch.softmax(logits.float() / options.temperature, dim=-1)
    probs, ids = probs.sort(descending=True)
    probs[(probs.cumsum(-1) - probs) >= options.top_p] = 0
    pick = torch.multinomial(probs, 1, generator=rng)
    return int(ids[pick].item())


class Text:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        chars = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        self.bytes = {chr(b): b for b in chars}
        for i, b in enumerate(b for b in range(256) if b not in chars):
            self.bytes[chr(256 + i)] = b
        self.special = set(tokenizer.all_special_ids)
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token):
        if token in self.special:
            return ""
        value = self.tokenizer.convert_ids_to_tokens(token)
        return self.decoder.decode(bytes(self.bytes[c] for c in value), final=False)

    def finish(self):
        return self.decoder.decode(b"", final=True)


class Engine:
    def __init__(self, device="auto", dtype=None, context=1024):
        self.device, self.dtype = device_info(device, dtype)
        path = MODEL_DIR
        if not (path / "revision.txt").exists() or (path / "revision.txt").read_text().strip() != REVISION:
            path = download()
        c = json.loads((path / "config.json").read_text())
        if (c["model_type"], c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]) != ("qwen3", 1024, 28, 16, 8, 128):
            raise ValueError("Only the official Qwen3-0.6B configuration is supported")
        if c["attention_bias"] or not c["tie_word_embeddings"] or c["rope_scaling"] or c["use_sliding_window"] or c["hidden_act"] != "silu":
            raise ValueError("Unsupported Qwen configuration")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        g = json.loads((path / "generation_config.json").read_text())
        self.stop = set(g["eos_token_id"])
        weights = {}
        for file in sorted(path.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name in weights:
                        raise ValueError(f"Duplicate weight: {name}")
                    weights[name] = f.get_tensor(name).to(device=self.device, dtype=self.dtype)
        if "lm_head.weight" in weights:
            if not torch.equal(weights["lm_head.weight"], weights["model.embed_tokens.weight"]):
                raise ValueError("The checkpoint's tied embedding and output weights differ")
            del weights["lm_head.weight"]
        expected = {"model.embed_tokens.weight", "model.norm.weight"}
        names = ["input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        for i in range(c["num_hidden_layers"]):
            expected.update(f"model.layers.{i}.{name}.weight" for name in names)
        if set(weights) != expected:
            raise ValueError(f"Unexpected weights; missing {expected - set(weights)}, extra {set(weights) - expected}")
        self.model = Model(c, weights, self.device, self.dtype, context)
        self.context = context

    def prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        if len(prompt) > self.context * 32:
            raise ValueError("Prompt text is too large")
        prompt.encode("utf-8")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )

    @torch.inference_mode()
    def generate(self, ids, options=None, use_cache=True, stop=True):
        options = options or Options()
        options.validate(len(ids), self.context)
        cache = self.model.cache() if use_cache else None
        rng = torch.Generator(device=self.device).manual_seed(options.seed)
        decoder = Text(self.tokenizer)
        tokens = list(ids)
        for i in range(options.max_tokens):
            current = tokens if cache is None or i == 0 else tokens[-1:]
            x = torch.tensor([current], device=self.device)
            logits = self.model.forward(x, cache)[0, -1]
            token = sample(logits, options, rng)
            tokens.append(token)
            reason = "stop" if stop and token in self.stop else "length" if i + 1 == options.max_tokens else None
            text = decoder.push(token)
            if reason:
                text += decoder.finish()
            yield {"token": token, "text": text, "finish_reason": reason}
            if reason:
                break
