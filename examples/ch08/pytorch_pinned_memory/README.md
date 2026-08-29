# PyTorch Pinned Memory Transfer Benchmark

This example compares host-to-device transfers from ordinary pageable CPU memory and page-locked (pinned) CPU memory.

## Why pinned memory exists

CPU memory is normally pageable: the operating system can move its physical pages while managing system RAM. A CUDA transfer from pageable memory may therefore require an intermediate page-locked staging buffer before the GPU can read the data efficiently.

Pinned memory keeps the selected host pages resident. CUDA can use that buffer as a reliable source or destination for DMA transfers. Pinned memory is still **CPU memory**, not VRAM, and it is a limited system resource.

PyTorch exposes this through `pin_memory()` and `pin_memory=True` in data loaders:

```python
pinned_batch = cpu_batch.pin_memory()
gpu_batch = pinned_batch.to("cuda", non_blocking=True)
```

The `non_blocking=True` argument can make the copy asynchronous with respect to the host when the source is suitable pinned memory and the surrounding stream structure allows overlap. It does not make the data transfer free, and it does not guarantee a speedup for every tensor size.

## What the program measures

For each configured size, the program measures:

1. transfer from pageable CPU memory with the ordinary blocking path;
2. transfer from pinned CPU memory with `non_blocking=True`, followed by an explicit synchronization;
3. the one-time cost of preparing the pinned buffer, printed separately from the repeated-transfer timing.

The benchmark repeats each transfer after a warm-up. The input data and the pinned buffer are allocated before timing, so the transfer table answers a specific question: **once a pinned buffer is already available, is repeated host-to-device movement faster?** The preparation time is shown separately because pinning a new batch for every single transfer can remove the benefit.

The result depends on size and platform. Small transfers can be dominated by Python, allocation, launch, and synchronization overhead. Larger repeated transfers are more likely to expose the benefit of a page-locked source buffer. On some systems the difference is small, and on a one-off transfer the pinning preparation cost may make the pinned path slower overall.

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

Run the default sizes, from 1 MiB to 256 MiB:

```bash
.venv/bin/python examples/ch08/pytorch_pinned_memory/pinned_memory_transfer.py
```

Use smaller sizes for a quick first run:

```bash
.venv/bin/python examples/ch08/pytorch_pinned_memory/pinned_memory_transfer.py \
  --sizes-mib 1,4,16,64 \
  --iterations 10
```

## How to interpret the output

Compare the `pageable_ms` and `pinned_nonblocking_ms` columns as the transfer size grows. Also record `pin_prepare_ms` and `break-even iters`:

- if `pinned_nonblocking_ms` is lower for repeated transfers, pinned memory is helping the transfer path;
- if both times are similar for small buffers, fixed overhead dominates the measurement;
- if `pin_prepare_ms` is large compared with the number of transfers, creating a pinned buffer for one use may not be worthwhile;
- a real input pipeline can reuse pinned buffers and overlap transfer with computation, which is a different experiment from a single isolated copy.

The script calculates the break-even point as:

```text
break-even iterations = pin_prepare_ms / (pageable_ms - pinned_nonblocking_ms)
```

This is the approximate number of transfers needed to recover the one-time pinning cost. For example, on the course RTX 5060 Ti with five measured iterations:

| Size | Pageable | Pinned | Pin preparation | Pinned speedup | Break-even |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 0.192 ms | 0.127 ms | 173.714 ms | 1.52x | 2672 iters |
| 4 MiB | 0.480 ms | 0.383 ms | 2.717 ms | 1.25x | 28 iters |
| 16 MiB | 1.687 ms | 1.407 ms | 9.173 ms | 1.20x | 33 iters |
| 64 MiB | 7.465 ms | 5.224 ms | 34.172 ms | 1.43x | 16 iters |
| 256 MiB | 28.616 ms | 20.667 ms | 138.737 ms | 1.38x | 18 iters |

For five transfers of 256 MiB, the pageable path takes approximately `5 x 28.616 = 143.080 ms`. The pinned path takes approximately `138.737 + 5 x 20.667 = 242.072 ms`, because the one-time preparation cost is not yet recovered. After roughly 18 transfers, the faster repeated pinned copies compensate for that initial cost. The 1 MiB preparation value is unusually high because first-use page-locking and allocator initialization can dominate such a small measurement; repeat the experiment and treat the first small-size result cautiously.

Every transfer is synchronized before its timing is reported. Without that boundary, Python could measure only submission time while the GPU copy is still in progress.
