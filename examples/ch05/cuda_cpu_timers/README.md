# CUDA CPU Timers

This example measures a CUDA vector-add kernel with a CPU wall-clock timer.

CUDA kernel launches are asynchronous with respect to the host. The example therefore calls `cudaDeviceSynchronize()` or `torch.cuda.synchronize()` immediately before starting the timer and immediately after launching the workload. The second synchronization is essential: without it, the timer would mostly measure launch and submission time.

## C++

```bash
cd examples/ch05/cuda_cpu_timers
nvcc cuda_cpu_timers.cu -O3 -o cuda_cpu_timers
./cuda_cpu_timers
./cuda_cpu_timers 16777216
```

## Python

From the repository root:

```bash
.venv/bin/python examples/ch05/cuda_cpu_timers/cuda_cpu_timers.py
.venv/bin/python examples/ch05/cuda_cpu_timers/cuda_cpu_timers.py --elements 16777216
```

## Interpretation

The reported interval is an end-to-end host observation of one submitted GPU operation. It includes CPU-side launch overhead and the time spent waiting at synchronization. It is appropriate when the question is the cost visible to the host, but CUDA events are more appropriate when the goal is to isolate the GPU-side interval.

