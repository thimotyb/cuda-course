#!/usr/bin/env python3
"""Compare a synchronized CPU timer with PyTorch CUDA event timing."""

import argparse
import time

import torch


def require_cuda() -> torch.device:
    """Stop early instead of silently turning the GPU benchmark into a CPU run."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this PyTorch installation.")
    return torch.device("cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU and CUDA event timers.")
    parser.add_argument("--elements", type=int, default=1 << 24)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.elements <= 0 or args.iterations <= 0:
        raise SystemExit("elements and iterations must be positive")

    # Select the device explicitly so the benchmark output and tensor placement
    # remain reproducible on machines that expose more than one GPU.
    device = require_cuda()
    # Allocate outside the timed loop so allocation is not mixed with kernel timing.
    # Allocation is outside the timed loop. This isolates the cost of the
    # element-wise operation from allocator and initialization effects.
    a = torch.ones(args.elements, device=device)
    b = torch.full_like(a, 2.0)

    # Warm up the CUDA context and the operation before collecting measurements.
    # Warm-up gives the CUDA runtime and PyTorch backend a chance to initialize
    # before the measurements are collected.
    _ = a + b
    torch.cuda.synchronize(device)

    cpu_times = []
    event_times = []
    for _ in range(args.iterations):
        # Python continues immediately after a CUDA launch, so synchronize both
        # boundaries when using a host wall clock.
        torch.cuda.synchronize(device)
        # perf_counter() measures host elapsed time. The synchronization before
        # it ensures earlier GPU work cannot contaminate this interval.
        cpu_start = time.perf_counter()
        result = a + b
        torch.cuda.synchronize(device)
        cpu_times.append((time.perf_counter() - cpu_start) * 1000.0)

        # Events are recorded on the current CUDA stream and measure its device
        # timeline. The stop event is synchronized before elapsed_time is read.
        # enable_timing=True asks PyTorch to create timestamp-capable CUDA events.
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = a + b
        stop_event.record()
        # The stop event is reached only after the addition in this stream has
        # completed, so elapsed_time() can now be read safely.
        stop_event.synchronize()
        event_times.append(start_event.elapsed_time(stop_event))

    torch.cuda.synchronize(device)
    # Validate the result after timing. Calling item() here is intentionally
    # outside the measured intervals because it synchronizes the host with CUDA.
    if result[0].item() != 3.0:
        raise RuntimeError("validation failed")

    cpu_ms = sum(cpu_times) / len(cpu_times)
    event_ms = sum(event_times) / len(event_times)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Elements: {args.elements}")
    print(f"Measured iterations: {args.iterations}")
    print(f"CPU wall-clock time: {cpu_ms:.3f} ms")
    print(f"CUDA event time: {event_ms:.3f} ms")
    print(f"CPU/event ratio: {cpu_ms / event_ms:.2f}x")
    print("Validation: OK")


if __name__ == "__main__":
    main()
