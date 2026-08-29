# PyTorch CUDA Memory Lab

This example makes PyTorch's CUDA memory behavior visible and connects it to a practical model-serving decision: how large a batch can fit in GPU memory?

## What the exercise demonstrates

### `memory_allocated()`

`torch.cuda.memory_allocated()` reports memory currently used by live PyTorch tensors. Creating a CUDA tensor normally increases this value. When the last Python reference to that tensor disappears, its storage becomes available for reuse by PyTorch.

### `memory_reserved()`

PyTorch uses a CUDA caching allocator. It asks CUDA for memory blocks and may keep unused blocks in a process-local cache so later tensor allocations can reuse them without asking the CUDA driver again. `torch.cuda.memory_reserved()` reports memory held by this allocator, so it is normally greater than or equal to `memory_allocated()`.

Reserved CUDA memory is **not pinned memory**. Reserved memory is GPU device memory kept by PyTorch for reuse. Pinned memory is page-locked CPU host memory used to make host-to-device and device-to-host transfers more efficient.

Typical behavior looks like this:

| Situation | `memory_allocated()` | `memory_reserved()` |
| --- | ---: | ---: |
| No tensors and empty cache | low or zero | low or zero |
| A live tensor exists | increases | increases, possibly by a larger block |
| Tensor is deleted | decreases | may remain high |
| `empty_cache()` is called | unchanged for live tensors | unused cached blocks may be released |

### `del` and `empty_cache()` are different

```python
del x
```

`del` removes a Python reference. If no other reference keeps the tensor alive, PyTorch can reuse its storage. It does not necessarily return the allocator's unused blocks to the CUDA driver.

```python
torch.cuda.empty_cache()
```

`empty_cache()` releases unused cached blocks from PyTorch's allocator back to the CUDA driver. It cannot free memory belonging to live tensors, model parameters, or active library state. It is therefore not a general solution for a real memory leak or an out-of-memory error caused by live objects.

The first part of the script shows this lifecycle by allocating two tensors, computing `c = a @ b`, deleting them one at a time, and finally calling `empty_cache()`. Peak counters are also printed because they preserve the maximum memory pressure reached during the experiment, even after cleanup.

### Finding a batch-size limit

The second part runs a deliberately simple workload:

```python
result = batch @ weights
```

The script doubles `batch` after every successful run. The input and output tensors grow with the batch size, just as activations and intermediate results often grow when a real model serves a larger batch. When the allocation no longer fits, PyTorch raises `torch.cuda.OutOfMemoryError`.

The largest successful batch is only an observation for this workload, GPU, data type, model shape, and current process state. A complete model also needs memory for weights, activations, temporary workspaces, and sometimes a cache such as an LLM KV cache. Leave headroom instead of treating the observed limit as a production setting.

## Requirements

- Python 3;
- PyTorch with CUDA support;
- an NVIDIA GPU visible to PyTorch.

From the repository root, create the shared environment if necessary:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
```

If `.venv` already exists, activate it:

```bash
source .venv/bin/activate
```

## Run

```bash
.venv/bin/python examples/ch09/pytorch_memory_lab/pytorch_memory_lab.py
```

Use smaller dimensions for a quick and safer first run:

```bash
.venv/bin/python examples/ch09/pytorch_memory_lab/pytorch_memory_lab.py \
  --lifecycle-size 1024 \
  --start-batch 128 \
  --max-batch 32768 \
  --features 512 \
  --outputs 512
```

Useful observations to record:

- how `memory_allocated()` changes when tensors become live or are deleted;
- whether `memory_reserved()` stays above allocated memory after deletion;
- how `empty_cache()` changes the reserved value;
- how peak memory grows as the batch doubles;
- why the largest successful batch is not automatically a safe production batch.

