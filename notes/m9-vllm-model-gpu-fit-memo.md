# Memo for M9.1.4: deciding whether an LLM fits a local GPU

The exercise should include a compact decision worksheet that students can use before launching a local model. The question is not only "does the model's parameter count fit in VRAM?" It is:

1. Do the model weights fit in the usable VRAM budget?
2. Is there enough memory left for the KV cache, temporary workspaces, CUDA graphs, and runtime overhead?
3. Does the GPU have enough memory bandwidth and compute capability for the intended latency and throughput?
4. Are the model architecture, data type, quantization format, driver, CUDA, PyTorch, and serving engine compatible?

## GPU-side values to collect

| GPU value | Why it matters | How to inspect it |
| --- | --- | --- |
| Total VRAM capacity | Hard upper bound for weights, KV cache, activations, workspaces, and runtime state | `nvidia-smi`, `torch.cuda.get_device_properties()` |
| Usable VRAM budget | vLLM may intentionally use only a fraction of total VRAM through `--gpu-memory-utilization` | `capacity * gpu_memory_utilization` |
| Memory bandwidth | Affects how quickly weights and KV data can be read; especially relevant during decode | GPU specification sheet |
| Compute capability | Determines supported CUDA kernels, Tensor Core paths, and data types | `torch.cuda.get_device_capability()` |
| Supported data types | BF16, FP16, FP8, INT8, or other formats change memory use and kernel availability | GPU documentation plus framework/model support |
| Current free VRAM | Other applications reduce the budget available to the server | `nvidia-smi` or `torch.cuda.mem_get_info()` |

The RTX 5060 Ti reference machine has 16 GB of GDDR7 and a specified peak memory bandwidth of 448 GB/s. These are different constraints: 16 GB answers how much state can fit, while 448 GB/s describes a limit on how quickly data can move through the memory subsystem. A model can fit in VRAM and still deliver disappointing latency if its workload is limited by memory traffic, kernel efficiency, or scheduling. See the [NVIDIA RTX 5060 family specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/).

## Model-side values to collect

| Model value | Why it matters |
| --- | --- |
| Parameter count | First approximation of weight storage |
| Weight data type | Bytes per parameter: FP32 = 4, FP16/BF16 = 2, INT8 = 1, approximately |
| Quantization format | Can reduce weights further, but scales, metadata, and kernel support add complexity |
| Number of layers | KV-cache storage grows with layer count |
| Number of KV heads | Grouped-query attention can use fewer KV heads than query heads, reducing KV-cache size |
| Head dimension | KV-cache storage grows with this dimension |
| Maximum context length | KV-cache storage grows with the number of cached tokens |
| Vocabulary and tied embeddings | Embedding and output weights contribute to the weight footprint |
| Runtime and serving configuration | Tensor parallelism, CUDA graphs, batch limits, and workspaces change actual usage |

The model configuration is the authoritative source for architectural values. For example, the official `Qwen/Qwen3-0.6B` configuration lists 28 hidden layers, 8 KV heads, head dimension 128, BF16 weights, and a maximum position length of 40960. See the [Qwen3-0.6B configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json).

## First estimate: weight memory

Use this approximation:

```text
weight memory ~= parameter count * bytes per parameter
```

For Qwen3-0.6B in BF16:

```text
0.6 billion parameters * 2 bytes ~= 1.2 GB ~= 1.12 GiB
```

This is only the weight estimate. It does not include the KV cache, temporary buffers, allocator fragmentation, CUDA graphs, tokenizer state, or vLLM runtime processes. A model that uses 14 GiB of weights is not automatically suitable for a 16 GiB GPU.

## KV-cache estimate

For a decoder-only transformer with standard K/V tensors, a useful first estimate is:

```text
KV bytes per token ~=
  2 * number of layers * number of KV heads * head dimension * bytes per KV value

total KV bytes ~= KV bytes per token * cached tokens * concurrent sequences
```

The first `2` accounts for one K tensor and one V tensor. For Qwen3-0.6B in BF16:

```text
2 * 28 layers * 8 KV heads * 128 values * 2 bytes
  = 114,688 bytes per token
  ~= 112 KiB per token
```

At 4096 cached tokens, this rough estimate is approximately 448 MiB per sequence. Eight concurrent sequences would therefore require approximately 3.5 GiB of KV cache before allocator and implementation overhead. The actual vLLM value must be measured because block allocation, attention implementation, prefix caching, graph capture, and scheduling policy affect the result.

## A practical fit worksheet

```text
usable VRAM
  = total VRAM * configured utilization
  - memory used by other processes

estimated required VRAM
  = weights
  + KV cache for the target context and concurrency
  + activations and temporary workspaces
  + runtime, CUDA graph, and allocator overhead

fit if estimated required VRAM < usable VRAM
```

Use a safety margin rather than targeting 100% of the budget. A first local test can start with a modest context and one request, then increase context length and concurrency while observing vLLM's reported KV-cache capacity and `nvidia-smi`.

## Bandwidth and performance expectations

VRAM capacity answers "can it run?"; bandwidth helps answer "how fast might it run?" Decode often revisits model weights and KV data for each generated token, so memory bandwidth can become important. A deliberately optimistic lower bound for reading 1.2 GB of weights at 448 GB/s is:

```text
1.2 GB / 448 GB/s ~= 2.7 ms
```

This is not a latency prediction. Real inference also includes non-ideal memory access, kernel work, attention and KV traffic, synchronization, launch overhead, batching, and Python/HTTP scheduling. Prefill and decode can have different bottlenecks: prefill often exposes larger matrix operations, while decode can be more sensitive to memory movement and small per-step work.

## Worked decision: Qwen3-0.6B on the 16 GB RTX 5060 Ti

For the course machine, the initial decision is favorable:

| Check | Approximate result | Decision |
| --- | ---: | --- |
| GPU capacity | 16 GiB class | Suitable for a small model |
| vLLM budget at `0.80` utilization | about 13 GiB | Leaves headroom for the host/runtime boundary |
| Qwen3-0.6B BF16 weights | about 1.12 GiB | Fits comfortably |
| KV cache at 4096 tokens, one sequence | about 0.44 GiB rough estimate | Fits comfortably |
| KV cache at 4096 tokens, eight sequences | about 3.5 GiB rough estimate | Still plausible, but measure it |
| Observed vLLM startup in the prototype | about 10.79 GiB KV cache available after loading | Confirms substantial remaining capacity on this setup |
| Peak memory bandwidth | 448 GB/s specified | Adequate for a useful local demo; not a guarantee of latency |

This is why `Qwen/Qwen3-0.6B` is an appropriate course model on the reference machine. The conclusion is not that every 0.6B model behaves identically, nor that any context/concurrency setting is safe. The final verification is always an actual server startup followed by a controlled request, memory observation, and context/concurrency experiment.

## Student-facing conclusion

> Choose a local LLM by checking weights, KV-cache demand, runtime overhead, supported data types, and the intended context/concurrency first. Use bandwidth and compute capability to reason about performance after the model fits. Then validate the estimate with the serving engine and a measured workload.
