#!/usr/bin/env python3
"""Measure a CUDA workload with a synchronized Python wall clock."""

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CUDA work with a CPU timer.")
    parser.add_argument("--elements", type=int, default=1 << 24)
    args = parser.parse_args()

    # Fail explicitly instead of accidentally benchmarking a CPU fallback.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this PyTorch installation.")

    # Making the device explicit keeps tensor placement visible in the example.
    device = torch.device("cuda:0")
    # Keep allocation outside the interval so the example measures the operation.
    # These allocations happen before timing. Including them would answer a
    # different question: how expensive is allocation plus computation?
    a = torch.ones(args.elements, device=device)
    b = torch.full_like(a, 2.0)

    # Warm up CUDA before measuring the steady-state operation.
    # The first CUDA operation may initialize the context or select a kernel.
    # Synchronizing after it makes the following measurement steady-state.
    _ = a + b
    torch.cuda.synchronize(device)

    # A CUDA operation returns before the GPU necessarily finishes it. The first
    # synchronization makes the CPU timer start at a clean device boundary.
    torch.cuda.synchronize(device)
    # perf_counter() runs on the CPU. The first synchronization establishes the
    # start boundary because Python can otherwise get ahead of the GPU.
    start = time.perf_counter()
    result = a + b

    # The second synchronization makes the elapsed time include completed GPU work.
    torch.cuda.synchronize(device)
    # This synchronization is essential. Without it, the CPU clock would stop
    # after enqueueing the operation and would underreport GPU execution time.
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Reading one scalar also verifies that the asynchronous operation completed
    # and produced the expected result.
    if result[0].item() != 3.0:
        raise RuntimeError("validation failed")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Elements: {args.elements}")
    print(f"CPU wall-clock interval: {elapsed_ms:.3f} ms")
    print("The interval includes Python launch and synchronization overhead.")


if __name__ == "__main__":
    main()
