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

The token is only needed for gated or private models. `VLLM_WSL2_ENABLE_PIN_MEMORY=1` enables vLLM's WSL2 pinned-memory path explicitly. The script also applies this value by default. It mounts the Hugging Face cache, exposes port 8001, enables all visible GPUs, limits the initial context to 4096 tokens, and reserves 80% of GPU memory for vLLM.

The script waits for `http://localhost:8001/health` before returning. Model download and server startup can take time on the first run.

## Send an inference request

In a second terminal:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py
```

The client prints end-to-end latency, the finish reason, generated-token count, approximate generation throughput, and the response text. Thinking output is disabled by default so that the small token budget produces a visible answer; pass `--enable-thinking` to expose Qwen3's reasoning output and its effect on the token budget. A finish reason of `length` means that `--max-tokens` was reached; `stop` means that the model ended the response normally.

## Capture a profiler trace

The launcher binds `reports/m9/vllm` on the host to `/reports` in the container and configures vLLM's PyTorch profiler. Profile one short request explicitly:

```bash
.venv/bin/python examples/ch09/vllm_docker_demo/client.py --profile --max-tokens 32
```

The client starts the profiling range before the request and stops it afterwards. vLLM then flushes trace files into `reports/m9/vllm`. Open the generated trace with [Perfetto](https://ui.perfetto.dev/) or another Chrome-trace-compatible viewer. Profiling adds significant overhead, so do not use `--profile` for normal throughput comparisons.

This produces PyTorch profiler traces, not `.nsys-rep` or `.ncu-rep` files. Nsight Systems and Nsight Compute require their CLI tools in the environment that launches the vLLM process and need a separate profiling image or command-line workflow.

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
docker stop m9-vllm-qwen
docker rm m9-vllm-qwen
```

The model remains in the mounted Hugging Face cache. Removing the container does not remove the downloaded model.

## Next extensions

The next version should add repeated requests, concurrent requests, prompt-length variation, latency percentiles, token throughput, memory sampling, and an optional Nsight Systems profiling mode. These measurements are more useful for the course than adding an orchestration framework at this stage.
