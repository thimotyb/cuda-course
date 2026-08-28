#!/usr/bin/env python3
"""Compare basic PyTorch tensor operations on CPU and CUDA.

The example is intentionally small and explicit. It shows that a tensor is not
only a block of numerical data: it also carries a device placement. That device
placement decides whether PyTorch uses CPU implementations or CUDA-backed GPU
implementations for supported operations.
"""

import argparse
import time

import torch


def require_cuda() -> torch.device:
    """Fail early if this exercise cannot really run on a CUDA GPU."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch. Install a CUDA-enabled PyTorch build "
            "and run on a machine with an NVIDIA GPU."
        )
    return torch.device("cuda:0")


def describe(name: str, tensor: torch.Tensor) -> None:
    """Print the tensor attributes that matter most for this module."""
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}"
    )


def tensor_workload(device: torch.device, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same tensor workload on whichever device is passed in.

    On CPU, this uses CPU tensor kernels. On CUDA, the allocations and supported
    operations are dispatched to GPU-backed implementations. Keeping the function
    identical for both devices makes the timing comparison easier to reason about.
    """
    a = torch.rand((size, size), device=device)
    b = torch.rand((size, size), device=device)

    # Element-wise operations are independent across tensor elements, which makes
    # them naturally data-parallel.
    mixed = (a + b) * 0.5

    # The @ operator is Python's matrix-multiplication operator. In PyTorch,
    # mixed @ b.T is equivalent to torch.matmul(mixed, b.T) for these 2D tensors.
    # b.T is the transpose of b, so both operands still have shape (size, size).
    # Matrix multiplication is much heavier than a simple element-wise operation;
    # for large enough inputs, this is the kind of workload that can amortize GPU
    # launch overhead and use optimized backend libraries such as cuBLAS.
    out = mixed @ b.T
    return mixed, out


def time_cpu_workload(size: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Measure the CPU version with a normal host-side wall clock."""
    start = time.perf_counter()
    mixed, out = tensor_workload(torch.device("cpu"), size)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return mixed, out, elapsed_ms


def time_cuda_workload(device: torch.device, size: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Measure the CUDA version with CUDA events.

    CUDA work is commonly asynchronous with respect to Python. CUDA events measure
    elapsed time on the GPU timeline, and the final synchronize waits until the
    queued GPU work has actually completed.
    """
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    mixed, out = tensor_workload(device, size)
    stop.record()
    torch.cuda.synchronize()

    return mixed, out, start.elapsed_time(stop)


def time_cuda_matmul(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Time one explicit CUDA matrix multiplication."""
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    out = a @ b
    stop.record()
    torch.cuda.synchronize()

    return out, start.elapsed_time(stop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic PyTorch tensor operations on CUDA.")
    parser.add_argument("--size", type=int, default=2048, help="Square matrix size for CPU/CUDA timing.")
    args = parser.parse_args()

    device = require_cuda()
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device} - {torch.cuda.get_device_name(device)}")

    # A tensor created from ordinary Python data starts on the CPU by default.
    data = [[1, 2], [3, 4]]
    x_cpu = torch.tensor(data, dtype=torch.float32)

    # Moving the tensor to CUDA performs a host-to-device transfer.
    x_cuda = x_cpu.to(device)
    describe("x_cpu", x_cpu)
    describe("x_cuda", x_cuda)

    # Factory functions can create tensors directly on the selected CUDA device.
    ones = torch.ones_like(x_cuda)
    random_values = torch.rand((2, 3), device=device)
    zeros = torch.zeros((2, 3), device=device)
    describe("ones", ones)
    describe("random_values", random_values)
    describe("zeros", zeros)

    # These familiar tensor operations stay on CUDA because their inputs are CUDA
    # tensors. PyTorch handles the low-level CUDA dispatch internally.
    grid = torch.ones((4, 4), device=device)
    grid[:, 1] = 0
    joined = torch.cat([grid, grid, grid], dim=1)
    squared = grid * grid
    gram = grid @ grid.T
    describe("grid", grid)
    describe("joined", joined)
    describe("squared", squared)
    describe("gram", gram)

    a = torch.rand((args.size, args.size), device=device)
    b = torch.rand((args.size, args.size), device=device)
    c, elapsed_ms = time_cuda_matmul(a, b)
    describe("c = a @ b", c)
    print(f"CUDA matmul time: {elapsed_ms:.3f} ms")

    # The same workload runs once on CPU and once on CUDA. The point is not to
    # claim that every GPU run is faster, but to show how to measure the boundary.
    print("\nCPU vs CUDA timing for the same tensor workload")
    print("workload: mixed = (a + b) * 0.5; out = mixed @ b.T")
    _, cpu_out, cpu_ms = time_cpu_workload(args.size)
    torch.cuda.synchronize()
    _, cuda_out, cuda_ms = time_cuda_workload(device, args.size)
    speedup = cpu_ms / cuda_ms if cuda_ms > 0 else float("inf")
    describe("cpu_out", cpu_out)
    describe("cuda_out", cuda_out)
    print(f"CPU workload time:  {cpu_ms:.3f} ms")
    print(f"CUDA workload time: {cuda_ms:.3f} ms")
    print(f"CUDA speedup over CPU for this workload: {speedup:.2f}x")

    # Reductions can produce CUDA tensors too. Calling .item() converts a one-value
    # CUDA tensor into a Python scalar, which requires the value to be available on
    # the host and can introduce synchronization.
    total = c.sum()
    print(f"sum(c) is a CUDA scalar before .item(): device={total.device}")
    print(f"sum(c).item() transfers the scalar result to Python: {total.item():.4f}")

    # PyTorch keeps a CUDA caching allocator. Reserved memory can be larger than
    # currently allocated tensor memory because cached blocks may be reused later.
    allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
    reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
    print(f"memory_allocated: {allocated_mb:.2f} MiB")
    print(f"memory_reserved: {reserved_mb:.2f} MiB")


if __name__ == "__main__":
    main()
