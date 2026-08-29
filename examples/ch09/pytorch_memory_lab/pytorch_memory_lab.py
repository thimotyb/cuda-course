#!/usr/bin/env python3
"""Make PyTorch's CUDA memory allocator visible and find a batch limit.

The example has two independent parts:

1. It observes memory while tensors are created, deleted, and released from
   PyTorch's cache.
2. It increases the batch size of a simple matrix multiplication until the
   workload no longer fits in the available GPU memory.

The measurements are intentionally small and controlled. They explain memory
behavior; they are not a replacement for a full model memory profile.
"""

from __future__ import annotations

import argparse
import gc

import torch


def mib(num_bytes: int) -> float:
    """Convert bytes to MiB for readable output."""
    return num_bytes / 1024**2


def require_cuda() -> torch.device:
    """Return cuda:0 or stop with a useful message on a CPU-only machine."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and "
            "make an NVIDIA GPU visible to the process."
        )
    return torch.device("cuda:0")


def print_memory(label: str, device: torch.device) -> None:
    """Print live allocation, allocator reservation, peaks, and driver space."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    print(f"\n{label}")
    print(f"  memory_allocated:     {mib(allocated):9.2f} MiB")
    print(f"  memory_reserved:      {mib(reserved):9.2f} MiB")
    print(f"  max_memory_allocated: {mib(peak_allocated):9.2f} MiB")
    print(f"  max_memory_reserved:  {mib(peak_reserved):9.2f} MiB")
    print(f"  device free / total:  {mib(free_bytes):9.2f} / {mib(total_bytes):.2f} MiB")


def clear_unreferenced_cuda_memory(device: torch.device) -> None:
    """Remove Python references and return unused allocator blocks to CUDA."""
    # gc.collect() deals with reference cycles that ordinary reference
    # counting cannot immediately reclaim. empty_cache() only releases blocks
    # that are already unused; it cannot destroy live tensors.
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def show_tensor_lifecycle(device: torch.device, size: int) -> None:
    """Show how allocated and reserved memory change during tensor lifetime."""
    print("\n=== Part 1: tensor lifetime and the CUDA caching allocator ===")
    clear_unreferenced_cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    print_memory("Initial state after empty_cache()", device)

    # These tensors are live Python objects whose storage resides in GPU VRAM.
    a = torch.rand((size, size), device=device)
    b = torch.rand((size, size), device=device)
    torch.cuda.synchronize(device)
    print_memory(f"After creating a and b ({size} x {size})", device)

    # Matrix multiplication creates another tensor and may use temporary
    # workspace. Peak counters record the highest pressure reached so far.
    c = a @ b
    torch.cuda.synchronize(device)
    print_memory("After c = a @ b", device)

    # del removes our Python reference. The storage becomes reusable by
    # PyTorch, but the allocator may keep the corresponding block reserved.
    del a
    torch.cuda.synchronize(device)
    print_memory("After del a", device)

    # Deleting all user tensors should reduce memory_allocated. It does not
    # promise that memory_reserved will return to zero.
    del b
    del c
    torch.cuda.synchronize(device)
    print_memory("After deleting b and c", device)

    # empty_cache() returns unused cached blocks to the CUDA driver. It does
    # not free memory belonging to live tensors or CUDA library state.
    torch.cuda.empty_cache()
    print_memory("After torch.cuda.empty_cache()", device)


def run_batch_workload(
    device: torch.device, batch_size: int, features: int, outputs: int
) -> None:
    """Run one batched matrix multiplication and wait for GPU completion."""
    # The input and output sizes grow with batch_size. This is a deliberately
    # simple proxy for a model whose activation memory grows with the batch.
    weights = torch.randn((features, outputs), device=device)
    batch = torch.randn((batch_size, features), device=device)
    result = batch @ weights
    torch.cuda.synchronize(device)

    # Keep result live until synchronization has completed, then release all
    # trial-local tensors before the next batch-size attempt.
    del result
    del batch
    del weights


def find_batch_limit(
    device: torch.device,
    start_batch: int,
    max_batch: int,
    features: int,
    outputs: int,
) -> None:
    """Double the batch size until the workload fails or max_batch is reached."""
    print("\n=== Part 2: finding a batch-size limit ===")
    print(
        f"Workload: batch @ weights, features={features}, outputs={outputs}, "
        f"start_batch={start_batch}, max_batch={max_batch}"
    )

    batch_size = start_batch
    last_successful = None
    while batch_size <= max_batch:
        clear_unreferenced_cuda_memory(device)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            run_batch_workload(device, batch_size, features, outputs)
        except torch.cuda.OutOfMemoryError:
            # A failed allocation can leave cached blocks behind, so clean up
            # before reporting the result or trying another independent run.
            clear_unreferenced_cuda_memory(device)
            print(f"  batch={batch_size:>9,}: OUT OF MEMORY")
            break

        peak = mib(torch.cuda.max_memory_allocated(device))
        reserved = mib(torch.cuda.memory_reserved(device))
        print(
            f"  batch={batch_size:>9,}: success, "
            f"peak_allocated={peak:,.2f} MiB, reserved_after_run={reserved:,.2f} MiB"
        )
        last_successful = batch_size
        batch_size *= 2

    if last_successful is None:
        print("No tested batch size fit. Reduce --start-batch or the matrix dimensions.")
    elif batch_size > max_batch:
        print(f"Maximum tested batch size: {last_successful:,} (the limit was not reached).")
    else:
        print(f"Largest successful batch size in this search: {last_successful:,}")

    print(
        "This is an observed limit for this workload, GPU, and process state; "
        "it is not a universal batch size for a complete model."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-size", type=int, default=2048)
    parser.add_argument("--start-batch", type=int, default=1024)
    parser.add_argument("--max-batch", type=int, default=1_048_576)
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--outputs", type=int, default=1024)
    args = parser.parse_args()

    device = require_cuda()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print("Reserved CUDA memory is cached GPU memory, not pinned host memory.")

    show_tensor_lifecycle(device, args.lifecycle_size)
    find_batch_limit(
        device,
        args.start_batch,
        args.max_batch,
        args.features,
        args.outputs,
    )


if __name__ == "__main__":
    main()
