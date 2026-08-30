# CUDA GPU Timers

This example compares two ways to measure the same CUDA workload:

- a CPU wall-clock timer, with explicit synchronization around the kernel;
- CUDA events recorded on the GPU timeline.

The workload is a simple element-wise vector addition. The computation is intentionally familiar so that the focus stays on asynchronous execution and timing boundaries.

The dedicated CPU-timer examples are in `../cuda_cpu_timers`. They isolate the host wall-clock method used in the CPU timer section. This directory contains the side-by-side comparison and the CUDA event implementation.

## Requirements

- NVIDIA GPU and working CUDA toolkit
- `nvcc` for the C++ example
- PyTorch with CUDA support for the Python example

## C++ example

From the repository root:

```bash
cd examples/ch05/cuda_gpu_timers
nvcc cuda_gpu_timers.cu -O3 -o cuda_gpu_timers
./cuda_gpu_timers
```

Optional arguments are the number of elements and the number of measured iterations:

```bash
./cuda_gpu_timers 16777216 20
```

## Python example

From the repository root, using the course virtual environment:

```bash
.venv/bin/python examples/ch05/cuda_gpu_timers/cuda_gpu_timers.py
.venv/bin/python examples/ch05/cuda_gpu_timers/cuda_gpu_timers.py --elements 16777216 --iterations 20
```

## What to observe

The CPU timer synchronizes before starting and after the kernel launch. It therefore measures completed GPU work, but it also includes host-side launch and synchronization overhead. The CUDA event interval starts and ends on the device stream, so it isolates the GPU-side workload more closely.

The first iteration is used as a warm-up. The reported values are averages over the remaining iterations. On a very small workload, the difference between the two measurements can be comparable to the kernel itself. For a larger workload, the device interval should dominate more of the total cost.

Neither measurement includes allocations performed before the timed loop. This is deliberate: allocation cost is a separate performance question and should be measured in a separate interval when needed.
