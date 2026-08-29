# From Microbenchmarks to Real Workloads

This example applies the measurement ideas from M8 to a small application-style PyTorch workload. It deliberately compares two different questions:

- **Microbenchmark:** how fast is a matrix operation when all tensors are already on the selected device?
- **Application benchmark:** how long does a complete input-to-output iteration take, including host preparation, transfers, computation, and result handling?

## What the example demonstrates

The script first warms up both workloads, then records repeated steady-state samples. It reports:

- mean latency;
- `p50`, `p95`, and `p99` latency;
- iterations per second;
- mean time spent preparing input, copying host-to-device, computing, and copying device-to-host.

The warm-up is excluded from the reported samples because the first iterations can include CUDA context setup, allocator initialization, kernel selection, and other one-time work. A production benchmark should report warm-up separately from steady state.

The application workload creates a CPU input, optionally pins it, transfers it to the selected device, executes `ReLU(inputs @ weights)`, copies the result back to the CPU, and reads one result value. Those operations represent the kinds of costs that a microbenchmark hides.

## Run

From the repository root:

```bash
.venv/bin/python examples/ch09/application_workload_benchmark/application_workload_benchmark.py
```

Force CPU or CUDA, or use a smaller first run:

```bash
.venv/bin/python examples/ch09/application_workload_benchmark/application_workload_benchmark.py --device cpu
.venv/bin/python examples/ch09/application_workload_benchmark/application_workload_benchmark.py --device cuda --batch-size 512 --features 512 --outputs 512
```

Compare a normal pageable input with an input pinned for each iteration:

```bash
.venv/bin/python examples/ch09/application_workload_benchmark/application_workload_benchmark.py --pinned
```

`--pinned` is intentionally a simple demonstration. It includes the cost of calling `pin_memory()` in input preparation. A real data pipeline should usually reuse pinned buffers, for example through `DataLoader(pin_memory=True)`, instead of pinning every batch from scratch.

## How to interpret the comparison

The microbenchmark can look much faster because its tensors and weights are already on the device. The application benchmark includes the costs needed by an actual request. The difference between their means is not automatically a GPU problem: it is the cost of orchestration and data movement that the microbenchmark excluded.

Use the stage means to identify the dominant part of the pipeline:

- large `input preparation` means CPU-side data creation is significant;
- large `host to device` or `device to host` means transfers are significant;
- large `compute` means the model-like operation dominates;
- a large difference between mean and `p95`/`p99` means occasional slow iterations affect tail latency.

Repeat the benchmark with larger `--batch-size` values. Fixed launch and synchronization costs are then spread over more work, while memory use and transfer sizes also grow. The goal is not to prove that CPU or GPU always wins, but to identify the workload size and bottleneck where the decision changes.

