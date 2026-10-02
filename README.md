<div align="center">

# AUREI

**A small language model engine, built to understand what happens inside.**

Python · PyTorch · Qwen3 · FastAPI

Runs locally on a laptop with 4GB of GPU memory.

[Try it](#try-it) · [How it works](#how-it-works) · [Verification](#checks-run-during-development) · [Results](#actual-results)

</div>

## Why I built this

I wanted to build the part that actually runs a language model: load its weights, process a prompt, choose the next token and keep going. AUREI is that project.

It uses the existing weights of [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). I wrote the model execution and generation loop in PyTorch, then added a local API that can handle two requests together. Hugging Face provides the official tokenizer and the reference model used in the checks.

The published code includes the engine, a CLI and an API with streaming responses and continuous batching. The three Python files sit in one folder, so you can follow the code without jumping through a large package structure.

**15 checks passed during development.** The actual weights were compared against the reference on CUDA and CPU. The benchmark run covered **240 requests** and **30,720 generated tokens**. The results below were recorded locally; development scripts and raw reports are excluded from this repository.

## Try it

Open PowerShell in the project folder. If the environment is already installed, start here:

```powershell
.\.venv\Scripts\python.exe cli.py "Explain KV caching in simple words." --max-tokens 96
```

The answer appears as the model generates it. CUDA is selected when available; otherwise the engine uses CPU.

### First setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The setup installs the pinned dependencies and downloads the model. Weights take about 1.5GB. PyTorch and its package cache also need space, so allow several GB in total.

**Downloads stay inside the project folder.** On this machine that means E:, including the model, package caches and temporary files. The model is stored in `models/qwen3-0.6b`.

The CUDA setup uses PyTorch 2.8.0 with CUDA 12.8. It needs a compatible NVIDIA driver; a separate CUDA toolkit installation is unnecessary. For a CPU installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Cpu
```

The weights and tokenizer use revision `c1899de289a04d12100db370d81485cdf75e47ca`. Once downloaded, generation works offline. To resume an interrupted download:

```powershell
.\.venv\Scripts\python.exe cli.py --download
```

### A few things to try

```powershell
.\.venv\Scripts\python.exe cli.py "Write a short poem." --temperature 0.6 --top-p 0.95 --seed 42
.\.venv\Scripts\python.exe cli.py "Say hello." --device cpu --max-tokens 32
.\.venv\Scripts\python.exe cli.py "Explain attention." --no-cache
```

Temperature zero uses greedy decoding. A higher temperature with `top_p` enables sampling. The default total context is 1,024 tokens, counting both the formatted prompt and the requested output.

## Use the API

Start the server:

```powershell
.\.venv\Scripts\python.exe server.py
```

Leave that terminal open and visit [the interactive API page](http://127.0.0.1:8000/docs). You can send a prompt from there without writing a client.

To try it from a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$body = '{"prompt":"Explain KV caching.","max_tokens":64,"temperature":0,"seed":42}'
$reply = Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -ContentType "application/json" -Body $body
$reply.text
```

<table>
<tr><th>Endpoint</th><th>What it does</th></tr>
<tr><td><code>GET /health</code></td><td>Shows readiness, device, active requests, queue size and cache memory</td></tr>
<tr><td><code>POST /generate</code></td><td>Returns the finished text, token IDs, usage and finish reason</td></tr>
<tr><td><code>POST /stream</code></td><td>Sends the answer gradually through server sent events</td></tr>
<tr><td><code>DELETE /requests/{id}</code></td><td>Cancels a waiting or active request</td></tr>
</table>

Streaming starts with a `start` event containing the request ID. Each `token` event contains an ID and a piece of text. The `done` event includes usage and tells you whether generation stopped naturally or reached its output limit. Failures after streaming starts arrive as `error` events.

Some token events contain no text because a Unicode character is still being assembled. The decoder keeps those bytes until the character is complete.

Invalid input returns 422. Bodies above 128 KiB return 413. A full queue or unhealthy engine returns 503. Cancelling an unknown or finished request returns 404. Unknown request fields are rejected.

The server runs locally at `127.0.0.1`, using one process. Press Ctrl+C to stop it. If memory is tight, start with one active request:

```powershell
.\.venv\Scripts\python.exe server.py --active 1 --context 1024 --queue 8
```

## How it works

The engine follows the [official model configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) and the [Transformers 4.57.1 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py).

The tokenizer turns a prompt into token IDs using the official chat template with thinking disabled. The engine then looks up embeddings, runs 28 decoder layers and computes scores for the next token. Generation chooses a token, decodes its text and repeats.

<table>
<tr><th>Model detail</th><th>Value</th></tr>
<tr><td>Hidden size</td><td>1,024</td></tr>
<tr><td>Decoder layers</td><td>28</td></tr>
<tr><td>Vocabulary</td><td>151,936 tokens</td></tr>
<tr><td>Query heads</td><td>16</td></tr>
<tr><td>Key and value heads</td><td>8 each</td></tr>
<tr><td>Head size</td><td>128</td></tr>
<tr><td>Feed forward size</td><td>3,072</td></tr>
<tr><td>Rotary theta</td><td>1,000,000</td></tr>
</table>

Each layer uses RMSNorm, rotary positions, grouped query attention, residual connections and a SiLU gated feed forward block. Q and K also have their own normalization. Attention uses PyTorch SDPA. The output projection shares the embedding weights.

The head size comes from the configuration. With 16 heads of size 128, the query projection is 2,048 wide even though the hidden size is 1,024. That is one of the details the implementation checks explicitly.

Safetensors supplies the weights. The checkpoint contains both the embedding and output head tensors; loading verifies they match and keeps one copy. RMSNorm reductions and rotary angles use float32 even when the weights use float16.

### Remembering previous tokens

Prompt processing fills the KV cache. After that, each step processes only the most recent token and reuses the previous keys and values. Without caching, the whole sequence is processed again at every step.

The normal generation path computes vocabulary scores only for the last valid position. The correctness checks can request scores for every position.

Generation stops at either official end token, `151645` or `151643`, or at the requested output limit. If the prompt plus reserved output does not fit the context, the engine rejects it rather than cutting it silently.

Text decoding keeps incomplete UTF8 characters between tokens. Hindi, Chinese, emoji and whitespace are checked against the official decoder, including every token prefix. If an output limit cuts a character in half, the final flush uses the same replacement character behavior as the official tokenizer.

### Handling two requests together

One worker thread owns inference. By default, two requests can be active and eight more can wait. Each active request gets its own slot in a fixed cache pool.

```text
K and V shape: [28 layers, 2 slots, 8 heads, 1024 positions, 128 dimensions]
```

The two cache tensors use **224 MiB** at float16 with two slots, or 112 MiB with one. Weights and temporary tensors need additional memory. Reserving the cache at startup makes its capacity predictable.

When a slot is available, the worker admits a waiting request and processes its prompt. It then advances all active requests together. Finished or cancelled requests release their slots after the current model step, allowing new requests to join.

Each slot tracks its own sequence length. Attention masks exclude padding, future positions and unused cache cells. A released slot is cleared before reuse. Each request also has its own sampling generator and text decoder.

Every request has an output queue limited to 32 events. A consumer that falls behind is cancelled with an error. Completion, disconnection, cancellation and shutdown all release request state. A model failure stops the worker and fails outstanding requests. After a GPU memory failure, restart with a smaller context or fewer active requests.

Prompt processing happens one request at a time, so admitting a long prompt briefly pauses active decoding. It is a deliberate limit of this small scheduler.

## Find your way around

<table>
<tr><th>File</th><th>Start here for</th></tr>
<tr><td><a href="engine.py">engine.py</a></td><td>Weight loading, model math, caching, sampling and decoding</td></tr>
<tr><td><a href="cli.py">cli.py</a></td><td>Trying a prompt in the terminal</td></tr>
<tr><td><a href="server.py">server.py</a></td><td>The API and request scheduler</td></tr>
<tr><td><a href="setup.ps1">setup.ps1</a></td><td>Installing the environment and downloading the model</td></tr>
<tr><td><a href="requirements.lock">requirements.lock</a></td><td>The exact dependency versions used in setup</td></tr>
</table>

## Checks run during development

Verification used tiny deterministic weights for 15 focused checks, followed by numerical comparisons using the actual model weights. The checks covered caching, padding, sequence lengths, stopping, context limits, cancellation, queue limits, request isolation, text decoding, shutdown and an injected memory failure.

The full comparison used the same weights, inputs and precision for the custom engine and Hugging Face. Four inputs included prompts with 128 and 512 tokens. Every prompt position and four cached decoding steps were compared. Cached versus uncached execution, mixed sequence lengths and the official tokenizer were checked too.

A live local API demonstration checked concurrent requests, streaming, socket disconnection, explicit cancellation, slot reuse and context rejection. These checks and their raw reports were run locally and are not included in this published runtime repository.

### What actually passed

The 15 quick checks and the live CUDA demonstration passed on this machine. All 20 direct float16 reference comparisons had zero logit difference in this run. The largest direct CPU float32 difference was **0.00007212**.

Different execution shapes can round differently. The largest float16 cached versus uncached difference was **0.146484375**, with RMS error **0.03089824**. The stricter float32 checks passed too. Exact agreement with a matching reference run does not mean every batching shape produces identical numbers.

Float32 uses absolute and relative tolerances of 0.0003. Direct float16 reference checks use 0.08 absolute and 0.01 relative tolerance. Checks across float16 execution shapes use 0.2 absolute, 0.02 relative and RMS error below 0.04.

The setup script was also rerun successfully. All 37 installed packages passed dependency checks. Automatic CPU fallback was tested with CUDA devices hidden.

## Actual results

These measurements were recorded on **2 October 2026**.

<table>
<tr><th>Hardware or software</th><th>Tested setup</th></tr>
<tr><td>GPU</td><td>NVIDIA RTX 3050 Laptop GPU, 4GB VRAM</td></tr>
<tr><td>CPU</td><td>Intel Core i7 12700H</td></tr>
<tr><td>Memory</td><td>Approximately 16GB RAM</td></tr>
<tr><td>Operating system</td><td>Windows 11</td></tr>
<tr><td>NVIDIA driver</td><td>577.00</td></tr>
<tr><td>Python</td><td>3.12.10</td></tr>
<tr><td>PyTorch</td><td>2.8.0 with CUDA 12.8</td></tr>
<tr><td>Precision</td><td>CUDA float16 and CPU float32</td></tr>
</table>

The following benchmark was run locally during development. Its scripts and raw CSV files are excluded from this repository.

Each request generates 128 tokens. Prompts contain 128 or 512 tokens, including the chat template. Concurrency is one or two, with two warmup rounds and ten measured rounds for each configuration. End tokens are ignored only in this benchmark so every run has the same output length.

The table compares the engine without caching, with caching, through the continuous batching API and against the Hugging Face SDPA reference. Throughput counts generated tokens across the whole workload. Time to first token includes prompt processing and waiting. P95 latency means 95% of measured requests completed within that time.

<table>
<tr><th>Method</th><th>Prompt tokens</th><th>Concurrency</th><th>Tokens/s</th><th>Mean first token, ms</th><th>P95 latency, s</th><th>Peak GPU memory, MiB</th></tr>
<tr><td>No cache</td><td>128</td><td>1</td><td>21.54</td><td>41.00</td><td>7.28</td><td>1168.07</td></tr>
<tr><td>No cache</td><td>128</td><td>2</td><td>20.26</td><td>3197.69</td><td>13.76</td><td>1168.07</td></tr>
<tr><td>No cache</td><td>512</td><td>1</td><td>6.39</td><td>127.77</td><td>20.18</td><td>1236.23</td></tr>
<tr><td>No cache</td><td>512</td><td>2</td><td>6.54</td><td>9920.34</td><td>39.96</td><td>1236.23</td></tr>
<tr><td>KV cache</td><td>128</td><td>1</td><td>23.54</td><td>43.18</td><td>7.11</td><td>1267.14</td></tr>
<tr><td>KV cache</td><td>128</td><td>2</td><td>25.65</td><td>2545.93</td><td>10.12</td><td>1267.14</td></tr>
<tr><td>KV cache</td><td>512</td><td>1</td><td>22.92</td><td>124.70</td><td>6.83</td><td>1320.78</td></tr>
<tr><td>KV cache</td><td>512</td><td>2</td><td>13.59</td><td>4751.84</td><td>21.35</td><td>1320.78</td></tr>
<tr><td>Continuous batching</td><td>128</td><td>1</td><td>18.87</td><td>63.29</td><td>8.91</td><td>1275.56</td></tr>
<tr><td>Continuous batching</td><td>128</td><td>2</td><td><strong>47.82</strong></td><td>104.14</td><td>7.02</td><td>1387.56</td></tr>
<tr><td>Continuous batching</td><td>512</td><td>1</td><td>16.10</td><td>131.71</td><td>8.38</td><td>1329.20</td></tr>
<tr><td>Continuous batching</td><td>512</td><td>2</td><td><strong>32.50</strong></td><td>233.11</td><td>9.26</td><td>1441.20</td></tr>
<tr><td>Hugging Face</td><td>128</td><td>1</td><td>18.29</td><td>45.90</td><td>9.69</td><td>1189.35</td></tr>
<tr><td>Hugging Face</td><td>128</td><td>2</td><td>20.37</td><td>3261.92</td><td>15.16</td><td>1189.35</td></tr>
<tr><td>Hugging Face</td><td>512</td><td>1</td><td>19.78</td><td>123.21</td><td>9.34</td><td>1271.41</td></tr>
<tr><td>Hugging Face</td><td>512</td><td>2</td><td>19.23</td><td>3412.16</td><td>15.36</td><td>1271.41</td></tr>
</table>

Caching helped most with the longer prompt. At concurrency one, throughput went from 6.39 to 22.92 tokens per second. Continuous batching reached 47.82 tokens per second for shorter prompts with two active requests. Its peak allocated GPU memory stayed below 1,442 MiB in these workloads.

The results also show the cost of serving. With one request, the HTTP path was slower than the direct cached engine. The cached concurrency two run had substantial variation; those observations remain in the report.

### How to read these numbers

The serial methods process simultaneous arrivals in order, so their concurrency two timings include queue time. Continuous batching is measured through real local HTTP streaming. The other methods run directly in Python. That transport difference is part of this comparison.

Timing uses a monotonic clock with CUDA synchronization at measurement boundaries. Model loading and downloading are excluded. Generation and decoding are included. P95 uses linear interpolation. GPU memory means peak PyTorch allocated memory, including weights and cache; it excludes driver overhead and other applications. The CSV also records reserved memory.

This is a short repeated workload on one laptop. The methods ran in a fixed order, and the GPU reached 86°C during the uncached workload. Heat, power settings and background activity can affect the results. These observations do not prove a general advantage over other inference libraries.

The fixed cache gives predictable capacity but uses more memory than the reference's growing cache. Aggregate results, individual request timings, exact workloads and environment records were saved locally.

## One improvement I measured

The profiler showed that computing vocabulary scores for every prompt position was expensive. For a prompt with 512 tokens, the multiplication `[512, 1024] × [1024, 151936]` took **13.190 ms** of GPU time, about 11.17% of the reported operator time.

Generation only needs scores for the last position. The engine now selects that hidden state before final normalization and output projection.

Across 20 alternating measurements, median prompt processing time fell from **121.534 ms to 110.529 ms**, a **9.06% reduction**. Median peak allocated memory fell from **1,302.15 MiB to 1,208.78 MiB**. The comparison checks that the final scores agree. This measures prompt processing on this setup, rather than an overall generation speedup.

The comparison saved raw measurements, a profiler table and a Chrome trace locally. Those development artifacts are excluded from this repository.

## What this version supports

This version focuses on Qwen3 with 0.6B parameters, a total context of 1,024 tokens and at most two active requests. The validated paths are CUDA float16 and CPU float32 on the laptop above. The bfloat16 option exists but has no reported benchmark here.

There is no training, fine tuning, quantization, distributed execution, custom CUDA kernel, paged cache or shared prompt cache. Prompt processing is not split into chunks. Longer contexts, more than two active requests, other GPUs, other operating systems and production deployment are unverified.

Batching and floating point rounding can change token choices when scores are close. Separate seeds keep request sampling state isolated, but identical seeds do not guarantee identical output across hardware and attention backends.

The memory failure test injects an exception rather than exhausting the GPU. The slow consumer check exercises the bounded output queue, rather than a sustained network stress test.

The small model can still give wrong facts, make arithmetic mistakes or leave answers incomplete. The checks establish that the engine executes the model correctly; they do not establish that its answers are true.
