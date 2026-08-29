#!/usr/bin/env python3
"""Compare pageable and pinned CPU memory for host-to-device transfers."""

from __future__ import annotations

import argparse
import time

import torch


def require_cuda() -> torch.device:
    """Select cuda:0 and stop clearly when this process cannot use CUDA."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch. This exercise requires a "
            "CUDA-enabled build and a visible NVIDIA GPU."
        )
    return torch.device("cuda:0")


def transfer_once(source: torch.Tensor, device: torch.device, non_blocking: bool) -> None:
    """Copy one CPU tensor to the GPU and wait until the copy is complete."""
    destination = source.to(device, non_blocking=non_blocking)
    # The host timer must include completed device work, not only enqueue time.
    torch.cuda.synchronize(device)
    del destination


def average_transfer_ms(
    source: torch.Tensor, device: torch.device, non_blocking: bool, iterations: int
) -> float:
    """Return the average end-to-end time of repeated CPU-to-GPU copies."""
    # Warm up the path so first-use CUDA setup does not dominate the result.
    transfer_once(source, device, non_blocking)

    start = time.perf_counter()
    for _ in range(iterations):
        transfer_once(source, device, non_blocking)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / iterations


def prepare_pinned(source: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Create a pinned copy and return it with the preparation time."""
    start = time.perf_counter()
    # pin_memory() allocates page-locked host storage and copies the data into
    # it. This one-time preparation is deliberately measured separately.
    # Docs: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html
    pinned = source.pin_memory()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return pinned, elapsed_ms


def parse_sizes(value: str) -> list[int]:
    """Parse a comma-separated list of positive MiB values."""
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes-mib",
        type=parse_sizes,
        default=[1, 4, 16, 64, 256],
        help="comma-separated transfer sizes in MiB",
    )
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    device = require_cuda()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Iterations per case: {args.iterations}")
    print("The pinned-buffer preparation cost is reported separately.")

    print(
        f"\n{'Size (MiB)':>12} {'pageable_ms':>15} "
        f"{'pinned_nonblocking_ms':>23} {'pin_prepare_ms':>17} "
        f"{'pinned speedup':>16} {'break-even iters':>18}"
    )
    print("-" * 110)

    for size_mib in args.sizes_mib:
        # Step 1: build a CPU tensor of exactly size_mib MiB (float32 elements).
        element_count = size_mib * 1024**2 // torch.tensor([], dtype=torch.float32).element_size()
        pageable = torch.randn(element_count, dtype=torch.float32)

        # Step 2: pin a copy of it and record the (one-time) pinning cost.
        pinned, pin_prepare_ms = prepare_pinned(pageable)

        # Step 3: time a blocking H2D copy from ordinary pageable memory. The
        # driver cannot DMA directly from pageable memory, so it first stages
        # the data into a temporary pinned buffer internally, then transfers it.
        pageable_ms = average_transfer_ms(pageable, device, False, args.iterations)

        # Step 4: time a non_blocking=True H2D copy from the already-pinned
        # buffer. Because the source is pinned, the driver can DMA straight
        # from host to device without that hidden staging copy.
        pinned_ms = average_transfer_ms(pinned, device, True, args.iterations)

        # Step 5: how many times faster the pinned path is per transfer.
        speedup = pageable_ms / pinned_ms if pinned_ms else float("inf")

        # Step 6: pinning is not free (Step 2's cost) — amortize it against the
        # per-transfer time saved (pageable_ms - pinned_ms) to see how many
        # repeated transfers of this buffer it takes to break even.
        per_transfer_saving = pageable_ms - pinned_ms
        break_even = (
            pin_prepare_ms / per_transfer_saving
            if per_transfer_saving > 0
            else float("inf")
        )

        print(
            f"{size_mib:12d} {pageable_ms:15.3f} {pinned_ms:23.3f} "
            f"{pin_prepare_ms:17.3f} {speedup:15.2f}x {break_even:18.1f}"
        )

        # Release both pageable and pinned host buffers before the next size.
        del pageable
        del pinned


if __name__ == "__main__":
    main()
