# M9.1.4 vLLM Container and Profiling Scratchpad

Working notes for turning M9.1.4 into a runnable exercise for Python developers who use PyTorch or vLLM.

## Core decision

Run vLLM in the official Docker image instead of installing vLLM directly in the host Python environment.

This keeps vLLM, PyTorch, CUDA user-space libraries, and Python dependencies together in a reproducible environment. The host still provides the NVIDIA kernel driver and the container runtime integration needed to expose the GPU.

The official image is:

```text
vllm/vllm-openai
```

Reference: <https://docs.vllm.ai/en/latest/deployment/docker/>

## Host requirements

The learner needs:

- A supported NVIDIA GPU with enough VRAM for the selected model.
- A working NVIDIA driver.
- Docker Engine or a compatible container engine.
- NVIDIA Container Toolkit, so Docker can pass the GPU into the container.
- Network access for the first model download, unless the model cache is already populated.
- A Hugging Face token for gated or private models.

The host does not need a vLLM installation. A local CUDA Toolkit is also not required merely to run the pre-built vLLM image, although it remains useful for the CUDA development and profiling exercises in the rest of the course.

Basic checks to include before the exercise:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:base-ubuntu22.04 nvidia-smi
```

The last command verifies that Docker can actually see the GPU. If it fails, installing Python packages inside the container will not solve the problem; the NVIDIA driver or Container Toolkit setup must be fixed first.

## Docker Desktop/WSL compatibility finding

On the course development machine, Docker Desktop with WSL 2 exposed the RTX 5060 Ti correctly. The first vLLM attempt with `vllm/vllm-openai:latest` (v0.28.0) failed during V1 engine initialization with `RuntimeError: UVA is not available`. The model was not too large and the GPU was visible; the failure occurred while creating a V1 unified virtual addressing buffer. The image also reported that `VLLM_USE_V1` was unknown, so this v0.28.0 image does not provide the older V0 fallback through that variable.

The current workaround is to set `VLLM_WSL2_ENABLE_PIN_MEMORY=1` inside the container and retest the V1 startup path. Native Linux hosts should test the default V1 path separately.

The WSL2 launcher should also set `VLLM_WSL2_ENABLE_PIN_MEMORY=1` before starting vLLM. This explicitly enables the pinned host-memory path used for CPU/GPU transfers in WSL2. Pinned memory is host memory and is distinct from the GPU memory reserved by PyTorch or vLLM.

## Prototype validation on the development machine

The Docker GPU integration was verified with `nvidia/cuda:12.8.1-base-ubuntu24.04` and `nvidia-smi`. The vLLM image then loaded Qwen3-0.6B successfully on the RTX 5060 Ti.

Observed vLLM startup values:

- model weights: 1.12 GiB of GPU memory;
- available KV cache: 10.79 GiB;
- configured context length: 4,096 tokens;
- estimated maximum concurrency at that context length: 24.67 requests;
- total GPU memory observed during startup: approximately 14.6 GiB.

The first prototype initially used host port 8000, but that port was already occupied by the course website. The launcher was changed to map host port 8001 to the container's port 8000. The site therefore remains on `http://localhost:8000`, while vLLM is exposed at `http://localhost:8001/v1`.

The first Python client request succeeded with 32 generated tokens in 448.04 ms, approximately 71.42 tokens/s. Qwen3 thinking output is disabled by default in the client so that a small token budget produces a visible answer; `--enable-thinking` re-enables it for a separate experiment.

## How to evaluate whether the model fits

The model choice should be justified by comparing model memory requirements with the GPU memory actually available, rather than by parameter count alone.

The values observed on the course machine were:

| Value | Measurement or estimate |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Total VRAM | 16,311 MiB, approximately 16 GB |
| VRAM already in use | 672 MiB |
| VRAM available before starting vLLM | 15,379 MiB |
| Qwen3 parameter count | 0.6 billion parameters |
| Qwen3 data type | BF16, 2 bytes per parameter |
| Approximate raw weight size | 0.6B x 2 bytes = 1.2 GB, approximately 1.1 GiB |

The raw weight estimate is only the first check. vLLM also needs memory for the model runtime, CUDA and PyTorch workspaces, temporary buffers, tokenizer and scheduler state, and the KV cache. The KV cache grows with context length, generated tokens, and concurrent requests. Qwen3-0.6B advertises a context length of up to 32,768 tokens, but the first lab should use a smaller limit so that the available memory margin remains visible and predictable.

This comparison shows that Qwen3-0.6B is comfortably below the available VRAM on this machine. It leaves several gigabytes for runtime allocations, KV cache, and profiling overhead. The conclusion is therefore based on:

1. Actual free VRAM from `nvidia-smi`.
2. The model's parameter count and storage data type.
3. The difference between raw weights and the full inference memory footprint.
4. The expected effect of context length and concurrency on KV-cache memory.
5. A safety margin for vLLM, CUDA, and profiling tools.

For the first run, use conservative limits such as `--max-model-len 4096` and `--gpu-memory-utilization 0.80`. Increase context length and concurrency only after observing memory use. A model can fit with one short request and still run out of memory when long contexts or many simultaneous sequences enlarge the KV cache.

The parameter count, BF16 data type, and maximum context length come from the [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). The VRAM values are local measurements from `nvidia-smi` and must be repeated on each learner's machine.

## Minimal server command

Use a small model for the first run. The image exposes an OpenAI-compatible server on port 8000:

```bash
export HF_TOKEN=your_token

docker run --rm \
  --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

For a repeatable course exercise, replace `latest` with a tested image tag after the vLLM version and GPU compatibility have been verified.

The cache mount avoids downloading the same model for every container invocation. A separate named Docker volume may be preferable in a multi-user or production setup.

The endpoint is:

```text
http://localhost:8000/v1
```

## Python client exercise

The client can run on the host or in a separate Python environment. It should use the OpenAI-compatible API rather than importing vLLM internals.

Possible dependencies:

```bash
python3 -m venv .venv-m9
.venv-m9/bin/python -m pip install openai
```

Minimal client shape:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-token",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Explain GPU memory in one sentence."}],
    max_tokens=64,
)

print(response.choices[0].message.content)
```

The model name in the request must match the model identifier served by vLLM. The exercise should record prompt tokens, generated tokens, elapsed time, and derived tokens per second.

## Benchmark progression

The exercise should progress in small, observable steps:

1. Start the container and send one request.
2. Repeat the same request several times and distinguish first-request startup effects from steady-state latency.
3. Send requests with different prompt lengths and generation limits.
4. Send several requests concurrently from Python.
5. Compare single-request latency, aggregate throughput, GPU utilization, and VRAM use.
6. Relate the results to batching, scheduling, memory capacity, and KV-cache growth.

Avoid making a generic claim that vLLM is always faster. The useful result is a measured explanation of when a serving engine improves a shared workload.

## Profiling strategy

Containerization does not prevent GPU profiling. CUDA activity generated by vLLM remains visible to NVIDIA tools, but the profiling process must be able to observe the process inside the container.

Recommended division of responsibilities:

- `nvidia-smi` on the host: quick GPU utilization, memory, and process observation.
- Nsight Systems CLI (`nsys`) in the vLLM container: end-to-end timeline, CUDA API calls, kernel launches, synchronization, idle periods, and process activity.
- Nsight Systems UI on the host: open the `.nsys-rep` report written to a mounted directory.
- Nsight Compute CLI (`ncu`) in the container: detailed analysis of selected kernels, not the complete serving workload.
- Nsight Compute UI on the host: open the `.ncu-rep` report.

NVIDIA's current Nsight Systems documentation recommends making the CLI available inside the container for containerized workloads. The report can then be shared with the host through a bind mount. See <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>.

Nsight Compute is appropriate for selected kernels and metrics. It can require access to GPU performance counters and can add substantial overhead, so it should be used with a short, controlled request and a narrow kernel filter. See <https://docs.nvidia.com/nsight-compute/NsightComputeCli/>.

## Profiling container design

The stock vLLM image may not contain the desired Nsight CLI tools. Two implementation options need to be evaluated:

### Option A: derived profiling image

Create a small Dockerfile based on the pinned vLLM image and install the matching Nsight CLI packages. This is the cleanest teaching setup because the command and its dependencies are versioned together.

Conceptual shape:

```dockerfile
FROM vllm/vllm-openai:<tested-tag>

# Install only the Nsight CLI packages required by the exercise.
# Exact package names depend on the base distribution and CUDA repository.
```

The final implementation must verify that the Nsight version, CUDA user-space libraries, host driver, and GPU architecture work together.

### Option B: separate profiler environment

Keep the vLLM image unchanged and use a profiling-capable CUDA image or host installation to collect data from the container. This reduces image customization but is more sensitive to process visibility, injection libraries, permissions, and version mismatches.

Prefer Option A for a self-contained lab if the resulting image remains practical to build and distribute.

## Report output mount

Use a host directory for reports, for example:

```bash
mkdir -p reports/m9
```

The final profiling command should write reports under a mounted path such as `/reports` inside the container. The host can then open the generated report with `nsys-ui` or `ncu-ui`.

Do not put large model files or profiler reports into Git. Keep the report directory ignored or document it as a local exercise output directory.

The first working prototype binds the host directory `reports/m9/vllm` to `/reports` and configures vLLM's PyTorch profiler with an absolute output path. Running the client with `--profile` calls `/start_profile`, sends one request, then calls `/stop_profile`. On the development machine this produced a compressed PyTorch trace (`rank0...pt.trace.json.gz`, approximately 1.7 MB) and `profiler_out_0.txt` in the host directory. The trace can be opened in <https://ui.perfetto.dev/>.

## Container permissions and limitations

The GPU must be exposed with `--gpus all` or the equivalent runtime configuration. Docker's default security profile can block `perf_event_open`, which affects some Nsight Systems CPU sampling features. `nsys status --environment` should be used to check the environment before profiling.

Do not begin the lab with broad privileged-container settings. Add only the capability or seccomp adjustment required by the selected profiling mode, and document the security trade-off if the exercise needs it.

Other practical limitations:

- Profiling can perturb latency and throughput, so profiled numbers must not be compared directly with normal benchmark numbers.
- vLLM may use multiple processes, CUDA graphs, fused kernels, and asynchronous scheduling; the first profile should focus on the system timeline rather than one presumed kernel.
- A single short request may not exercise continuous batching. Concurrency is needed to expose the serving engine's scheduling behavior.
- Nsight Compute can replay or serialize work and is therefore unsuitable for measuring production-like throughput.
- GPU performance-counter access may depend on driver policy, operating system, container security, or administrator settings.

## Implementation checklist

- [ ] Choose and test a pinned `vllm/vllm-openai` image tag.
- [ ] Choose a small default model that fits the course GPU.
- [ ] Add Docker and NVIDIA Container Toolkit requirements to M9.1.4.
- [ ] Add a GPU visibility check before starting the server.
- [ ] Add the minimal server command and explain the cache mount.
- [ ] Add a Python client example using the OpenAI-compatible API.
- [ ] Add single-request and concurrent-request benchmark scripts.
- [ ] Decide whether to provide a derived image containing `nsys` and `ncu`.
- [ ] Test report collection and host-side UI opening.
- [ ] Test `nvidia-smi`, Nsight Systems, and selected Nsight Compute workflows on the target GPU.
- [ ] Document profiler permissions and the difference between profiling data and benchmark data.
- [ ] Add only stable, verified instructions to the learner-facing module after the implementation is tested.

## Teaching exercise design: profiler-led efficiency lab

The useful demonstration is not simply showing that the GPU is busy. The exercise should compare equivalent workloads under different serving configurations and use profiling data to explain why one configuration is more efficient.

The comparison must keep the following variables controlled:

- same GPU;
- same model and data type;
- same total number of input and output tokens;
- same context and generation limits unless context length is the variable under test;
- separate warm-up from steady-state measurements;
- separate normal benchmark runs from profiler runs.

The target conclusion is not that one setting is always faster. It should explain which bottleneck is dominant and why a particular configuration improves throughput, latency, or memory capacity for the tested workload.

### Experiment 1: single request versus concurrent requests

Run the same workload with concurrency levels such as `1`, `4`, `8`, and `16`. Measure:

- time to first token (TTFT);
- time per output token (TPOT);
- inter-token latency (ITL);
- end-to-end latency per request;
- aggregate tokens per second;
- GPU memory and KV-cache usage;
- request completion rate and P99 latency.

This experiment makes batching, scheduling, latency hiding, and the throughput/latency trade-off visible. A single request may leave parts of the GPU underused, while concurrent requests expose more independent work to the engine. Higher concurrency can eventually increase queueing, memory pressure, and tail latency.

Prefer `vllm bench serve` with streaming measurements when the installed image provides it. It reports TTFT, TPOT, ITL, throughput, and latency percentiles more precisely than a client that measures only the complete HTTP response. Repeating a benchmark against the same server can reuse prefix-cache entries and inflate throughput, so restart the server, vary the seed, or otherwise control cache reuse between comparable runs.

### Experiment 2: short context versus long context

Keep the generated-token limit fixed and vary the input context, for example:

```text
128 tokens
1024 tokens
4096 tokens
```

Observe the difference between:

- prefill: processing the input prompt;
- decode: generating output tokens autoregressively;
- KV-cache allocation and growth;
- latency to the first token;
- steady-state token generation speed.

This connects memory capacity and locality to a real LLM workload. Longer contexts increase the amount of initial work and the memory required for attention state. A model can fit comfortably for one short request and still approach the memory limit when context length or concurrency grows.

### Experiment 3: CUDA Graphs versus eager execution

Compare the default vLLM configuration with eager execution:

```bash
# Default configuration: CUDA Graphs are used when supported.
vllm serve Qwen/Qwen3-0.6B ...

# Eager execution: skip CUDA Graph capture.
vllm serve Qwen/Qwen3-0.6B --enforce-eager ...
```

Measure startup time separately from steady-state serving. Use Nsight Systems to inspect:

- CUDA API calls and kernel-launch overhead;
- the number and grouping of kernel launches;
- synchronization points;
- GPU idle intervals;
- CUDA Graph execution;
- the cost of compilation, warm-up, and graph capture.

The expected trade-off is faster startup and simpler development behavior with eager execution, versus better steady-state decode efficiency when CUDA Graphs can be reused. The result depends on the model, request shapes, and GPU.

### Nsight Systems: application and engine timeline

Nsight Systems is the first profiler to use because it provides the whole execution timeline. Profile a short, controlled request window rather than an entire long-running server. For vLLM, the current documentation recommends options such as:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  vllm serve Qwen/Qwen3-0.6B ...
```

For multiprocessing, test `VLLM_WORKER_MULTIPROC_METHOD=spawn` as recommended by the vLLM profiling documentation. The CLI must be available in the environment that runs the vLLM process. Save the report to a directory mounted from the container and open it with the host-side Nsight Systems UI.

The first analysis should identify whether time is dominated by:

- Python and HTTP overhead;
- prompt prefill;
- autoregressive decode;
- CUDA launch and synchronization overhead;
- memory transfers;
- GPU idle time;
- batching or queueing;
- compilation and warm-up.

### Nsight Compute: selected kernel analysis

Nsight Compute should be used on a short, controlled capture of one or a few representative kernels, not to measure production-like server throughput. It can replay or serialize work and can add substantial overhead.

Possible kernels to investigate include:

- GEMM or SGEMM kernels used by linear layers;
- FlashAttention kernels;
- normalization kernels;
- KV-cache update or attention kernels.

The useful metrics are:

- achieved occupancy;
- register usage;
- shared-memory usage;
- DRAM throughput;
- L2 cache hit rate;
- SM utilization;
- warp stall reasons.

These metrics connect the serving workload to the earlier modules:

| Earlier concept | What the profiler can reveal in the vLLM lab |
| --- | --- |
| M3: SMs, warps, occupancy | Active warps, achieved occupancy, warp stalls, and latency hiding |
| M4: locality and memory hierarchy | DRAM traffic, L2 behavior, memory bandwidth, and attention/KV-cache movement |
| M5: performance measurement | Kernel duration, application timeline, warm-up, steady state, and percentiles |
| M6: transfers and pinned memory | Host/device copies and whether transfer or staging overhead is visible |
| M7: configuration and resources | Registers, shared memory, launch shape, CUDA Graph behavior, and resource limits |
| M8: PyTorch and CUDA runtime | CUDA memory use, streams, synchronization, and runtime API activity |

Pinned memory is relevant mainly to input and output transfers. After the model is loaded, vLLM keeps weights and the KV cache primarily in device memory, so the central optimization questions are normally batching, prefill/decode balance, KV-cache capacity, kernel efficiency, and scheduling.

### Recommended result table

The final exercise should produce a table such as:

| Experiment | Concurrency | Context | Mode | TTFT | TPOT | Throughput | VRAM |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Baseline | 1 | 512 | CUDA Graphs | ... | ... | ... | ... |
| Concurrent workload | 8 | 512 | CUDA Graphs | ... | ... | ... | ... |
| Long context | 8 | 4096 | CUDA Graphs | ... | ... | ... | ... |
| Eager execution | 8 | 512 | Eager | ... | ... | ... | ... |

The written conclusion should have the form: the selected configuration improves throughput or latency because it exposes more parallel work, reuses execution graphs, improves batching, or reduces a particular memory or scheduling bottleneck. It should also state the cost, such as higher VRAM use, longer queueing, slower startup, or worse P99 latency.

### Why LangChain is not part of the first lab

The first lab should use the direct OpenAI-compatible client. This keeps the endpoint, request shape, streaming behavior, latency, token counts, and server configuration visible. LangChain can be introduced later for chains, retrieval, tools, or application composition, after the GPU and serving behavior has been measured directly.
