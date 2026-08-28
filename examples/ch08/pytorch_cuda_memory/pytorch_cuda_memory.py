#!/usr/bin/env python3
"""Demonstrate PyTorch CUDA memory inspection tools.

PyTorch uses a CUDA caching allocator. That allocator is good for performance
because future tensor allocations can reuse memory blocks without asking the CUDA
driver every time. The consequence is that "memory used by live tensors" and
"memory reserved by the PyTorch process" are related but not identical.

Important distinction: PyTorch reserved CUDA memory is not pinned memory.
Reserved CUDA memory is GPU device memory kept by PyTorch for reuse. Pinned
memory is page-locked CPU host memory used for faster host-device transfers.
"""

import argparse

import torch


def mib(num_bytes: int) -> float:
    """Convert bytes to mebibytes for readable console output."""
    return num_bytes / 1024**2


def require_cuda() -> torch.device:
    """Select cuda:0 and stop early if PyTorch cannot use CUDA."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "torch.cuda.is_available() is False. This exercise requires a "
            "CUDA-enabled PyTorch build and a visible NVIDIA GPU."
        )
    return torch.device("cuda:0")


def print_memory(label: str, device: torch.device) -> None:
    """Print the memory counters students should learn to distinguish."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PyTorch CUDA memory counters.")
    parser.add_argument("--size", type=int, default=2048, help="Square tensor size.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print torch.cuda.memory_summary() at the end.",
    )
    args = parser.parse_args()

    device = require_cuda()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA runtime version: {torch.version.cuda}")
    print(f"Selected device: {device} - {torch.cuda.get_device_name(device)}")
    print("Reserved CUDA memory here means cached GPU memory, not pinned host memory.")

    # Reset peak statistics so the peak counters describe only this exercise.
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    print_memory("Start after empty_cache()", device)

    # Allocate two tensors directly on the GPU. memory_allocated should increase
    # because these tensors are live Python objects holding CUDA memory.
    a = torch.rand((args.size, args.size), device=device)
    b = torch.rand((args.size, args.size), device=device)
    torch.cuda.synchronize(device)
    print_memory("After allocating a and b", device)

    # Matrix multiplication creates a third CUDA tensor. It also creates temporary
    # internal work that can affect peak memory counters.
    c = a @ b
    torch.cuda.synchronize(device)
    print_memory("After c = a @ b", device)

    # Deleting one Python reference releases that tensor for PyTorch's allocator,
    # but reserved memory may stay high because the allocator can keep blocks for
    # later reuse.
    del a
    torch.cuda.synchronize(device)
    print_memory("After del a", device)

    # Delete the remaining tensors. In a simple tensor-only case, allocated
    # memory usually drops substantially here. After library-backed operations
    # such as matmul, some active internal allocations or workspaces may remain.
    del b
    del c
    torch.cuda.synchronize(device)
    print_memory("After deleting all tensors", device)

    # empty_cache releases unused cached blocks back to the CUDA driver. It does
    # not free memory still owned by live tensors or internal library state.
    torch.cuda.empty_cache()
    print_memory("After torch.cuda.empty_cache()", device)

    # Peaks remain useful after cleanup because they describe the maximum memory
    # pressure reached during the measured section.
    print("\nPeak counters describe the maximum pressure reached during the run.")

    if args.summary:
        print("\nCompact memory summary")
        print(torch.cuda.memory_summary(device=device, abbreviated=True))


if __name__ == "__main__":
    main()
