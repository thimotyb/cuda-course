#!/usr/bin/env python3
"""Inspect the most useful torch.cuda commands for a first CUDA workflow.

The goal is to make PyTorch's CUDA control surface visible. Most PyTorch users
do not launch CUDA kernels manually, but they still need to know how to inspect
devices, place tensors, measure GPU work, and reason about memory.
"""

import argparse

import torch


def require_cuda() -> torch.device:
    """Return cuda:0 and stop with a clear message if CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "torch.cuda.is_available() is False. PyTorch is installed, but this "
            "environment cannot currently see a CUDA-capable GPU."
        )
    return torch.device("cuda:0")


def print_runtime_info() -> None:
    """Print PyTorch/CUDA runtime information before touching a device."""
    print("Runtime")
    print(f"  torch.__version__: {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")


def print_device_info(device: torch.device) -> None:
    """Print the selected GPU and a few useful hardware properties."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)

    print("\nSelected device")
    print(f"  torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"  torch.cuda.get_device_name({index}): {torch.cuda.get_device_name(index)}")
    print(f"  compute capability: {props.major}.{props.minor}")
    print(f"  total memory: {props.total_memory / 1024**3:.2f} GiB")
    print(f"  multiprocessor count: {props.multi_processor_count}")


def print_memory(label: str, device: torch.device) -> None:
    """Show PyTorch CUDA allocator state in MiB."""
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"{label}: allocated={allocated:.2f} MiB, reserved={reserved:.2f} MiB")


def time_cuda_matmul(device: torch.device, size: int) -> float:
    """Time one matrix multiplication with CUDA events.

    CUDA operations are usually asynchronous from Python's point of view. Events
    are recorded into the CUDA stream, and synchronize() waits until the GPU has
    completed the work before the elapsed time is read.
    """
    a = torch.rand((size, size), device=device)
    b = torch.rand((size, size), device=device)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    c = a @ b
    stop.record()

    torch.cuda.synchronize(device)
    print(f"  result tensor device: {c.device}")
    print(f"  result tensor shape: {tuple(c.shape)}")
    return start.elapsed_time(stop)


def demonstrate_streams(device: torch.device, size: int) -> None:
    """Show the default stream and one explicitly-created stream.

    Streams are ordered queues of CUDA work. This example uses one custom stream
    for demonstration; it is not trying to prove overlap or improve performance.
    """
    print("\nStreams")
    print(f"  current stream: {torch.cuda.current_stream(device)}")
    print(f"  default stream: {torch.cuda.default_stream(device)}")

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        x = torch.rand((size, size), device=device)
        y = x.relu()

    # The host waits for work submitted to this stream before y is used below.
    stream.synchronize()
    print(f"  custom stream completed work for tensor on: {y.device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect useful torch.cuda commands.")
    parser.add_argument("--size", type=int, default=2048, help="Square matrix size for the CUDA timing.")
    args = parser.parse_args()

    print_runtime_info()
    device = require_cuda()

    # A device context is useful when code can run on more than one GPU. On a
    # single-GPU machine, cuda:0 is the only CUDA device visible to PyTorch.
    with torch.cuda.device(device):
        print_device_info(device)

        print("\nMemory")
        print_memory("  before allocation", device)
        work = torch.empty((args.size, args.size), device=device)
        print_memory("  after allocation", device)
        del work
        torch.cuda.empty_cache()
        print_memory("  after del + empty_cache", device)

        demonstrate_streams(device, min(args.size, 1024))

        print("\nCUDA event timing")
        elapsed_ms = time_cuda_matmul(device, args.size)
        print(f"  matrix multiplication time: {elapsed_ms:.3f} ms")

        # .item() moves one scalar value back to Python. This is useful for
        # logging, but it also creates a CPU-visible synchronization point.
        scalar = torch.ones((), device=device) * 7
        print(f"\nScalar before .item(): value is on {scalar.device}")
        print(f"Scalar after .item(): Python value is {scalar.item()}")


if __name__ == "__main__":
    main()
