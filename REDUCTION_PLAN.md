# Reduction Occupancy Example Plan

## Goal

Create a CUDA example that is more useful than `vector_add` for profiling and for teaching GPU compute architecture topics in M3.

The example should make Nsight Systems and Nsight Compute show meaningful information about:

- SM usage;
- thread blocks;
- warps;
- occupancy;
- shared memory;
- synchronization;
- launch configuration;
- kernel timing.

The proposed example is a parametrized vector reduction program named `reduction.cu`.

## Why Parallel Reduction

Parallel reduction is a good fit for M3 because it is simple enough to explain, but rich enough to exercise real GPU execution behavior.

It gives the course a kernel that:

- launches many blocks and many threads;
- uses `__syncthreads()`;
- uses shared memory;
- exposes block-size tradeoffs;
- creates useful Nsight Compute occupancy data;
- can be profiled with different launch configurations;
- does not require introducing matrix multiplication before M4.

## Target Location

```text
examples/ch03/reduction_occupancy/
```

Expected files:

```text
examples/ch03/reduction_occupancy/reduction.cu
examples/ch03/reduction_occupancy/README.md
```

## Functional Requirements

`reduction.cu` must:

- allocate a large host array of `float` values;
- use a default size of `N = 1 << 26`;
- initialize the input data with simple deterministic values;
- compute a CPU reference sum;
- allocate device memory for input data and block partial sums;
- copy input data from host to device;
- launch a CUDA reduction kernel that computes one partial sum per block;
- copy partial sums back to the host;
- complete the final reduction of partial sums on the CPU for the first version;
- compare GPU and CPU results with a reasonable floating-point tolerance;
- print a clear PASS/FAIL result.

The program output should include:

- `N`;
- block size;
- number of blocks;
- shared memory per block;
- number of iterations;
- CPU sum;
- GPU sum;
- absolute error;
- average kernel time;
- PASS/FAIL status.

Example output shape:

```text
N: 67108864
block size: 256
blocks: 131072
shared memory per block: 1024 bytes
iterations: 20
CPU sum: ...
GPU sum: ...
absolute error: ...
average kernel time: ... ms
PASS
```

## CUDA Requirements

The implementation must:

- use `blockIdx.x`, `blockDim.x`, and `threadIdx.x`;
- use shared memory for intra-block reduction;
- use `__syncthreads()` during reduction;
- handle input sizes that are not exact multiples of the block size;
- support configurable block size from the command line;
- support at least these block sizes: `64`, `128`, `256`, `512`, `1024`;
- reject invalid block sizes cleanly;
- compile with `nvcc`;
- be readable enough for students to inspect during the architecture module.

## Profiling Requirements

The program should produce useful profiling data rather than a tiny toy kernel.

It must:

- use a default input large enough to occupy the GPU meaningfully;
- support an `iterations` argument to make kernel runtime easier to profile;
- use a warmup launch before timed iterations;
- use CUDA events to measure average GPU kernel time;
- avoid making the profile dominated only by `cudaMalloc`, initialization, or tiny kernel launch overhead;
- compile with `-O3 -lineinfo` in the documented profiling workflow.

Default run configuration:

```text
N = 1 << 26
block_size = 256
iterations = 20
```

CLI proposal:

```bash
./reduction
./reduction 67108864 256 20
./reduction 67108864 64 20
./reduction 67108864 1024 20
```

## Suggested Profiling Commands

Build:

```bash
nvcc reduction.cu -O3 -lineinfo -o reduction
```

Run:

```bash
./reduction
./reduction 67108864 256 20
```

Nsight Systems:

```bash
nsys profile --trace=cuda --stats=true -o reduction-nsys ./reduction
```

Nsight Compute:

```bash
ncu --set full --force-overwrite -o reduction-ncu ./reduction
ncu-ui reduction-ncu.ncu-rep
```

## Teaching Experiments

Students should run the same program with different block sizes:

```bash
./reduction 67108864 64 20
./reduction 67108864 128 20
./reduction 67108864 256 20
./reduction 67108864 512 20
./reduction 67108864 1024 20
```

They should compare:

- average kernel time;
- number of blocks;
- theoretical occupancy;
- achieved occupancy;
- active warps;
- eligible warps;
- stall reasons;
- memory throughput;
- shared-memory behavior;
- whether block size changes improve or hurt runtime.

## Nsight Compute Sections To Discuss

The M3 lesson should guide students toward:

- Launch Statistics;
- Occupancy;
- Warp State Statistics;
- Scheduler Statistics;
- Memory Workload Analysis;
- Source Counters, if available;
- Speed Of Light throughput summary.

## First Implementation Scope

Start with one robust version:

- one reduction kernel;
- shared-memory reduction inside each block;
- one partial sum per block;
- CPU final sum of partial results;
- configurable `N`, block size, and iterations.

Do not add multiple kernel variants in the first pass unless the single-kernel version does not produce sufficiently useful profiler data.

Possible later extensions:

- add a naive reduction variant;
- add an optimized two-elements-per-thread variant;
- add a register-pressure variant to demonstrate occupancy cliffs;
- add a shared-memory-heavy variant;
- add a comparison table to M3 after collecting real profiler data.

## Integration Plan

After implementation:

1. Add `examples/ch03/reduction_occupancy/reduction.cu`.
2. Add `examples/ch03/reduction_occupancy/README.md`.
3. Update `examples/README.md`.
4. Add a short M3 guided exercise linking to the example.
5. Run:

```bash
python3 scripts/non_regression_guard.py check
```

6. If M3 locked text changes intentionally, update only the M3 lock.
