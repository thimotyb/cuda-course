# Bandwidth Bottleneck

This example makes a memory-access bottleneck visible with two CUDA copy kernels:

- `coalescedCopy`: adjacent threads access adjacent `float` values;
- `stridedCopy`: adjacent threads access values separated by a configurable stride.

Both kernels copy the same number of useful elements. The program reports CUDA-event kernel time and effective bandwidth. The strided kernel may generate more actual memory traffic than the useful bytes counted by the formula, because the memory transactions contain data that the kernel does not use.

## 1. Compile and run

From the repository root:

```bash
cd examples/ch06/bandwidth_bottleneck
nvcc bandwidth_bottleneck.cu -O3 -lineinfo -o bandwidth_bottleneck
./bandwidth_bottleneck
```

Arguments are:

```text
./bandwidth_bottleneck [useful_elements] [stride] [iterations]
```

For a shorter profiling run:

```bash
./bandwidth_bottleneck 16777216 16 5
./bandwidth_bottleneck 16777216 32 5
```

The effective bandwidth calculation is:

```text
effective bandwidth = (bytes read + bytes written) / kernel time
```

For this copy, useful traffic is `count * sizeof(float) * 2`. The strided case still uses that useful-traffic numerator, so a lower result exposes the cost of inefficient memory transactions.

## 2. Nsight Systems: application timeline

Nsight Systems shows when the CPU launches the kernels and when the GPU executes them.

```bash
nsys profile --trace=cuda --stats=true --force-overwrite=true \
  -o bandwidth-nsys ./bandwidth_bottleneck 16777216 16 5
nsys-ui bandwidth-nsys.nsys-rep
```

In the GUI:

1. Open the CUDA HW or CUDA API rows in the timeline.
2. Find the repeated `coalescedCopy` and `stridedCopy` launches.
3. Compare their GPU duration and the gaps between launches.
4. Check whether the CPU is waiting at synchronization points.
5. Use the timeline to confirm that the kernel interval, rather than allocation or initialization, is the interval being analyzed.

Nsight Systems is the right first tool for the end-to-end timeline. It does not replace the detailed memory counters for one kernel.

## 3. Nsight Compute: kernel and memory metrics

Profile the kernels with a short run:

```bash
ncu --set full --kernel-name regex:'^coalescedCopy' \
  --launch-skip 1 --launch-count 1 --force-overwrite -o bandwidth-coalesced-ncu \
  ./bandwidth_bottleneck 16777216 16 5
ncu --set full --kernel-name regex:'^stridedCopy' \
  --launch-skip 1 --launch-count 1 --force-overwrite -o bandwidth-strided-ncu \
  ./bandwidth_bottleneck 16777216 16 5
ncu-ui bandwidth-coalesced-ncu.ncu-rep
ncu-ui bandwidth-strided-ncu.ncu-rep
```

If a metric is unavailable on the installed GPU or driver, collect the report without `--set full` and select the available sections explicitly:

```bash
ncu --section MemoryWorkloadAnalysis --section SpeedOfLight \
  --kernel-name regex:'^coalescedCopy' \
  --launch-skip 1 --launch-count 1 --force-overwrite -o bandwidth-coalesced-ncu \
  ./bandwidth_bottleneck 16777216 16 5
ncu --section MemoryWorkloadAnalysis --section SpeedOfLight \
  --kernel-name regex:'^stridedCopy' \
  --launch-skip 1 --launch-count 1 --force-overwrite -o bandwidth-strided-ncu \
  ./bandwidth_bottleneck 16777216 16 5
```

In Nsight Compute, inspect:

- `Launch Statistics`: grid and block configuration;
- `GPU Speed Of Light Throughput`: memory-throughput pressure;
- `Memory Workload Analysis`: requested and actual global-memory traffic;
- `Memory Workload Analysis` efficiency metrics: how much requested traffic was served efficiently;
- `Source Counters`, when available: source-level evidence for the access pattern.

Compare the two kernels using the same useful element count. A strided kernel can show similar useful bytes but lower effective bandwidth and higher actual traffic. That gap is evidence of wasted bandwidth, not of less logical work.

## 4. Questions

- Does increasing `stride` reduce effective bandwidth?
- Which kernel has the larger difference between requested and actual memory traffic?
- Does the GPU timeline show a meaningful CPU-side gap between launches?
- Is the kernel limited by memory throughput, or is the workload too small to reach steady state?
- Which code or data-layout change would improve coalescing, and does a second measurement confirm the improvement?

## 5. Cleanup

```bash
rm -f bandwidth_bottleneck bandwidth-nsys.nsys-rep bandwidth-nsys.sqlite \
  bandwidth-coalesced-ncu.ncu-rep bandwidth-strided-ncu.ncu-rep
```
