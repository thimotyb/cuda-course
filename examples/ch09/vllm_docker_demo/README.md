# vLLM Docker Demo

This first M9 prototype starts Qwen3-0.6B in the official vLLM Docker image and sends one request from Python through the OpenAI-compatible API.

The demo deliberately uses the direct OpenAI client instead of LangChain. LangChain is useful when the lesson is about chains, tools, retrieval, or application composition. It would hide the serving endpoint and add dependencies before the course has measured the GPU workload itself.

## Requirements

- Linux with an NVIDIA GPU and a working NVIDIA driver;
- Docker Engine or Docker Desktop with GPU access;
- NVIDIA Container Toolkit on a native Linux host;
- Docker access from the current user;
- Python 3 and the `openai` package;
- enough VRAM for Qwen3-0.6B and the selected context/concurrency settings.

Check the Docker GPU integration before starting vLLM:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Install the Python client dependency

From the repository root:

```bash
.venv/bin/python -m pip install -r examples/ch09/vllm_docker_demo/requirements.txt
```

## Start vLLM

From the repository root:

```bash
export HF_TOKEN=your_token
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
bash examples/ch09/vllm_docker_demo/start_vllm.sh
```

The token is only needed for gated or private models. `VLLM_WSL2_ENABLE_PIN_MEMORY=1` enables vLLM's WSL2 pinned-memory path explicitly. The script also applies this value by default. It mounts the Hugging Face cache, exposes port 8001, enables all visible GPUs, limits each request to a 4096-token maximum sequence length, and reserves 80% of GPU memory for vLLM.

The launcher enables Qwen3 tool calling with `--enable-auto-tool-choice --tool-call-parser hermes`. This is needed by Open WebUI because it can send `tool_choice="auto"`, even for an ordinary chat prompt. If you change `MODEL` to a model that needs another parser, override it with `TOOL_CALL_PARSER=...`; check the vLLM tool-calling documentation for the supported parser. After changing this setting, recreate the vLLM container:

```bash
bash examples/ch09/vllm_docker_demo/stop_vllm.sh
bash examples/ch09/vllm_docker_demo/start_vllm.sh
```

The script waits for `http://localhost:8001/health` before returning. Model download and server startup can take time on the first run.
After changing scheduler settings, wait until the launcher prints `vLLM is ready.` before running the client. vLLM may spend extra time compiling kernels, autotuning, and capturing CUDA graphs after a restart.

## Use a web console

Open WebUI provides a browser-based chat interface for the OpenAI-compatible vLLM endpoint. Start vLLM first, then launch Open WebUI from the repository root:

```bash
bash examples/ch09/vllm_docker_demo/start_vllm.sh
bash examples/ch09/vllm_docker_demo/start_open_webui.sh
```

The console is available at [http://localhost:3000](http://localhost:3000). The launcher connects Open WebUI to the vLLM endpoint at `http://host.docker.internal:8001/v1`, so it works with the default port used by this demo on Linux and Docker Desktop. The first visit asks you to create a local Open WebUI account; the account and chat history are stored in the persistent Docker volume `m9-open-webui-data`.

Stop the web console without stopping vLLM with:

```bash
bash examples/ch09/vllm_docker_demo/stop_open_webui.sh
```

Open WebUI is only a frontend: prompts sent from the browser still pass through the same vLLM server, scheduler, KV cache, and GPU that the Python client exercises.

## Send an inference request

In a second terminal:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py
```

The client prints end-to-end latency, the finish reason, generated-token count, approximate generation throughput, and the response text. Thinking output is disabled by default so that the small token budget produces a visible answer; pass `--enable-thinking` to expose Qwen3's reasoning output and its effect on the token budget. A finish reason of `length` means that `--max-tokens` was reached; `stop` means that the model ended the response normally.

## Send concurrent inference requests

Exercise 1.4 studies serving behavior under load, so the client can submit several requests at the same time:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --requests 8 --concurrency 4 --max-tokens 64
```

The client prints aggregate throughput, mean latency, `p50`, `p95`, `p99`, generated tokens, and finish reasons. By default it prints only the summary; add `--show-responses` when you need to inspect every answer.
Before sending requests, the client waits up to 120 seconds for `http://localhost:8001/health`; change this with `--wait-timeout` if the server is still warming up.

Use streaming mode when you also want time-to-first-token and inter-token latency:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --requests 8 --concurrency 4 --max-tokens 64 --stream
```

Keep the prompt, request count, concurrency, and `--max-tokens` fixed while comparing server configurations. For example, restart vLLM with different scheduler limits:

With `vllm/vllm-openai:latest` reporting vLLM 0.28.0, the launcher leaves `--max-num-seqs` and `--max-num-batched-tokens` unset. For this image, the scheduler defaults are:

```text
max_num_seqs = 128
max_num_batched_tokens = 2048
```

The restricted runs below deliberately tighten those limits. They are not expected to improve every metric; they create a constrained scheduler configuration that can expose degraded behavior under concurrent load. If the limit becomes the bottleneck, expect lower throughput, higher queueing or tail latency, and Perfetto traces with smaller or more fragmented execution ranges instead of larger batched GPU work.

```bash
bash examples/ch09/vllm_docker_demo/stop_vllm.sh
MAX_NUM_SEQS=1 bash examples/ch09/vllm_docker_demo/start_vllm.sh
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --requests 8 --concurrency 4 --max-tokens 64

bash examples/ch09/vllm_docker_demo/stop_vllm.sh
MAX_NUM_SEQS=4 bash examples/ch09/vllm_docker_demo/start_vllm.sh
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --requests 8 --concurrency 4 --max-tokens 64

bash examples/ch09/vllm_docker_demo/stop_vllm.sh
MAX_NUM_BATCHED_TOKENS=1024 bash examples/ch09/vllm_docker_demo/start_vllm.sh
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --requests 8 --concurrency 4 --max-tokens 64
```

Use the non-profiled runs for the main latency and throughput table. Profiling adds overhead, so capture only one representative concurrent run after you know which configuration you want to inspect.

## Capture a profiler trace

The launcher binds `reports/m9/vllm` on the host to `/reports` in the container and configures vLLM's PyTorch profiler. Profile one short request explicitly:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --profile --max-tokens 32
```

Profile a concurrent run with the same mechanism:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --requests 8 --concurrency 4 --max-tokens 64 --profile
```

The client starts the profiling range before the request batch and stops it afterwards. vLLM then flushes trace files into `reports/m9/vllm`. Open the generated trace with [Perfetto](https://ui.perfetto.dev/) or another Chrome-trace-compatible viewer. Profiling adds significant overhead, so do not use `--profile` for normal throughput comparisons.

Stopping the profiler can take longer than the inference request because vLLM has to flush the PyTorch trace to disk. The client waits up to 180 seconds for `start_profile` and `stop_profile`; change this with `--profile-control-timeout` if a large trace needs more time. If a profiled run times out while stopping the profiler, do not immediately start a second profiled run. First clean up the profiler state:

```bash
curl -X POST http://localhost:8001/stop_profile
```

This produces PyTorch profiler traces, not `.nsys-rep` or `.ncu-rep` files. Nsight Systems and Nsight Compute require their CLI tools in the environment that launches the vLLM process and need a separate profiling image or command-line workflow.

## Inspect the trace in Perfetto

From the repository root, list the latest profiler outputs:

```bash
ls -lh reports/m9/vllm
```

Open the newest `rank0.*.pt.trace.json.gz` file in Perfetto. If the viewer does not accept the compressed file, decompress a copy first:

```bash
gzip -dk reports/m9/vllm/rank0.*.pt.trace.json.gz
```

Use Perfetto search to inspect these events:

- `execute_context`: main vLLM execution ranges. A first generation range corresponds to prompt prefill; repeated generation ranges correspond to autoregressive decode steps.
- `cutlass`: GEMM and matrix-multiplication kernels, often using Tensor Core paths. These connect the trace to tiled matrix multiplication, arithmetic intensity, and compute throughput.
- `flash_fwd`: FlashAttention kernels. These connect the trace to attention work, KV-cache reads, L2 reuse, and global-memory bandwidth.
- `reshape_and_cache`: KV-cache update kernels. These show where generated prompt or decode state is written into the cache.
- `Memcpy`: host-device or device-host copies. If these are much shorter than the compute kernels, transfers are not the dominant cost for this request.

Also inspect the text summary:

```bash
sed -n '1,120p' reports/m9/vllm/profiler_out_0.txt
```

Record at least:

- `Self CPU time total`;
- `Self CUDA time total`;
- the largest `execute_context` CUDA ranges;
- the dominant CUDA kernels by `Self CUDA`;
- the number of decode-generation calls;
- whether the request ended with `Finish reason: length` or `stop`.

For a short single-request baseline such as `--max-tokens 32`, a `Finish reason: length` is expected when the token budget cuts the answer off. This is acceptable for profiling because it keeps the captured range small and repeatable. In a concurrent trace, compare the spacing and duration of prefill, decode, `cutlass`, `flash_fwd`, `reshape_and_cache`, and `Memcpy` ranges with the single-request baseline. Look for larger scheduled batches, more overlapping request work, longer per-request latency, and whether memory-copy intervals remain small compared with compute kernels.

Try a different prompt or generation limit:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --prompt "What is a CUDA stream?" --max-tokens 32
```

## Observe the GPU

While the server is starting or serving requests, inspect the host GPU from another terminal:

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

This connects the application-level measurements to earlier modules: the container receives a GPU through the runtime, model weights and the KV cache consume device memory, and request concurrency changes the amount of parallel work available to the GPU.

## Stop the server

```bash
bash examples/ch09/vllm_docker_demo/stop_vllm.sh
```

The model remains in the mounted Hugging Face cache. Removing the container does not remove the downloaded model.

## Next extensions

The next version should add prompt-length variation, GPU memory sampling during the request batch, and an optional Nsight Systems profiling mode. These measurements are more useful for the course than adding an orchestration framework at this stage.
