#!/usr/bin/env python3
"""Compare a PyTorch microbenchmark with an application-style workload.

The microbenchmark keeps tensors on the selected device and measures only the
matrix operation. The application benchmark includes CPU input preparation,
host-to-device movement, model-like computation, device-to-host movement, and
result handling. Repeating both workloads makes warm-up and steady-state
latency visible.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def select_device(requested: str) -> torch.device:
    """Select the requested device, or choose CUDA when it is available."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work; this is a no-op for CPU measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    """Return a simple nearest-rank percentile in milliseconds."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def print_latency_summary(label: str, samples_ms: list[float]) -> None:
    """Print steady-state latency and throughput for one measured workload."""
    total_seconds = sum(samples_ms) / 1000.0
    print(f"\n{label}")
    print(f"  samples: {len(samples_ms)}")
    print(f"  mean:    {statistics.mean(samples_ms):8.3f} ms")
    print(f"  p50:     {percentile(samples_ms, 0.50):8.3f} ms")
    print(f"  p95:     {percentile(samples_ms, 0.95):8.3f} ms")
    print(f"  p99:     {percentile(samples_ms, 0.99):8.3f} ms")
    print(f"  throughput: {len(samples_ms) / total_seconds:8.2f} iterations/s")


def model_step(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Represent a small model layer with a regular GPU-friendly workload."""
    return torch.relu(inputs @ weights)


def run_microbenchmark(
    device: torch.device,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Measure only model computation after tensors are already on the device."""
    # This is a classic microbenchmark: data placement and output handling are
    # deliberately outside the timed region. It answers "how fast is this
    # operation when the application has already prepared everything?".
    for _ in range(warmup):
        output = model_step(inputs, weights)
    synchronize(device)
    del output

    samples_ms = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        output = model_step(inputs, weights)
        synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        del output
    return samples_ms


def run_application_benchmark(
    device: torch.device,
    batch_size: int,
    features: int,
    outputs: int,
    warmup: int,
    iterations: int,
    pinned: bool,
) -> tuple[list[float], dict[str, list[float]]]:
    """Measure a complete input-to-output pipeline and its major stages."""
    # We keep weights on the device, as a real model would after loading. Every
    # iteration creates a fresh CPU input and returns a CPU result, making data
    # movement and host-side handling part of the application measurement.
    weights = torch.randn((features, outputs), device=device)

    def one_iteration(collect_stages: bool) -> tuple[float, dict[str, float]]:
        stage_times = {"input preparation": 0.0, "host to device": 0.0, "compute": 0.0, "device to host": 0.0}

        start = time.perf_counter()

        # Stage 1 - input preparation: build a fresh batch on the CPU, the way
        # a real data-loading step would produce it. Optionally pin it here, so
        # the (one-time, per-buffer) pinning cost is charged to this stage
        # rather than hidden inside the transfer stage below.
        input_start = time.perf_counter()
        inputs_cpu = torch.randn((batch_size, features))
        if pinned:
            # pin_memory() is intentionally inside input preparation here. A
            # production DataLoader normally prepares and reuses such buffers.
            inputs_cpu = inputs_cpu.pin_memory()
        stage_times["input preparation"] = (time.perf_counter() - input_start) * 1000.0

        # Stage 2 - host to device: copy the input batch onto the compute
        # device. non_blocking is only honored by CUDA when the source is
        # pinned; synchronize() below makes the copy's true cost visible even
        # though the call itself may return before the copy is complete.
        transfer_start = time.perf_counter()
        inputs = inputs_cpu.to(device, non_blocking=pinned)
        synchronize(device)
        stage_times["host to device"] = (time.perf_counter() - transfer_start) * 1000.0

        # Stage 3 - compute: run the model step entirely on-device.
        compute_start = time.perf_counter()
        output = model_step(inputs, weights)
        synchronize(device)
        stage_times["compute"] = (time.perf_counter() - compute_start) * 1000.0

        # Stage 4 - device to host: bring the result back to CPU memory, as an
        # application would before returning or logging it.
        output_start = time.perf_counter()
        output_cpu = output.cpu()
        synchronize(device)
        stage_times["device to host"] = (time.perf_counter() - output_start) * 1000.0

        # Stage 5 - result handling: touch one value so this final step is
        # part of the measured workflow too, not just the raw data movement.
        _ = float(output_cpu[0, 0])
        del inputs_cpu, inputs, output, output_cpu
        return (time.perf_counter() - start) * 1000.0, stage_times

    for _ in range(warmup):
        one_iteration(collect_stages=False)

    samples_ms = []
    stages = {"input preparation": [], "host to device": [], "compute": [], "device to host": []}
    for _ in range(iterations):
        elapsed_ms, stage_times = one_iteration(collect_stages=True)
        samples_ms.append(elapsed_ms)
        for name, value in stage_times.items():
            stages[name].append(value)

    del weights
    return samples_ms, stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--outputs", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--pinned",
        action="store_true",
        help="pin each generated input before the application transfer",
    )
    args = parser.parse_args()

    if min(args.batch_size, args.features, args.outputs, args.warmup, args.iterations) <= 0:
        parser.error("batch size, dimensions, warmup, and iterations must be positive")

    device = select_device(args.device)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(
        f"Workload: batch={args.batch_size}, features={args.features}, "
        f"outputs={args.outputs}, warmup={args.warmup}, iterations={args.iterations}"
    )

    # Prepare one set of device-resident tensors for the microbenchmark. The
    # allocation and initial transfer are setup, not steady-state work.
    inputs = torch.randn((args.batch_size, args.features), device=device)
    weights = torch.randn((args.features, args.outputs), device=device)
    micro_samples = run_microbenchmark(device, inputs, weights, args.warmup, args.iterations)
    print_latency_summary("Microbenchmark: compute only", micro_samples)
    del inputs, weights

    app_samples, stages = run_application_benchmark(
        device,
        args.batch_size,
        args.features,
        args.outputs,
        args.warmup,
        args.iterations,
        args.pinned,
    )
    print_latency_summary("Application benchmark: end-to-end pipeline", app_samples)

    print("\nApplication stage means")
    for name, samples in stages.items():
        print(f"  {name:<20} {statistics.mean(samples):8.3f} ms")

    print("\nInterpretation")
    print("  Microbenchmark excludes placement and result handling.")
    print("  Application benchmark includes host preparation, transfers, compute, and output handling.")
    print("  Compare p95/p99 with the mean to see whether occasional slow iterations affect users.")
    print("  Repeat with a larger batch to test whether fixed overhead is amortized.")


if __name__ == "__main__":
    main()
