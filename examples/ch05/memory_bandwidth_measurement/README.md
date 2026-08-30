# Memory Bandwidth Measurement

This is the introductory M5 memory-movement exercise. It launches one simple CUDA copy kernel, measures it with CUDA events, and calculates effective bandwidth from useful bytes and kernel time.

The example deliberately does not compare coalesced and strided accesses. Those access-pattern concepts are introduced in M6.

## Compile and run

```bash
cd examples/ch05/memory_bandwidth_measurement
nvcc memory_bandwidth_measurement.cu -O3 -lineinfo -o memory_bandwidth_measurement
./memory_bandwidth_measurement 16777216 20
```

The output reports:

- the detected GPU memory clock and bus width;
- theoretical peak bandwidth calculated from those properties;
- kernel time measured on the GPU timeline;
- useful traffic: one read and one write of one `float` per element;
- effective bandwidth in decimal `GB/s`;
- the percentage of theoretical bandwidth reached by this kernel.

The program calculates the theoretical value at runtime:

```text
memory clock rate * (memory bus width / 8) * 2 / 10^9
```

On the reference RTX 5060 Ti, this is approximately `448 GB/s`. The effective value is normally lower because it depends on the specific kernel, workload, runtime overhead, and memory behavior.

## Nsight Systems

Use Nsight Systems to see the application timeline and distinguish allocation, initialization, event synchronization, and kernel launches:

```bash
nsys profile --trace=cuda --stats=true --force-overwrite=true \
  -o memory-bandwidth-nsys ./memory_bandwidth_measurement 16777216 5
nsys-ui memory-bandwidth-nsys.nsys-rep
```

In the timeline, locate `copyKernel`, compare its duration with the surrounding CUDA API calls, and identify the synchronization after each measured launch.

## Nsight Compute

Use Nsight Compute to inspect the single copy kernel:

```bash
ncu --set full --kernel-name regex:'^copyKernel' \
  --launch-skip 1 --launch-count 1 --force-overwrite \
  -o memory-bandwidth-ncu ./memory_bandwidth_measurement 16777216 5
ncu-ui memory-bandwidth-ncu.ncu-rep
```

Start with `Launch Statistics`, `GPU Speed Of Light Throughput`, and `Memory Workload Analysis`. At this stage, record the available memory-throughput metrics and compare the profiler's kernel duration with the program's CUDA-event duration. The goal is to learn where the measurements are found and how they relate, not yet to optimize the access pattern.

## Questions

- How many useful bytes are read and written per iteration?
- Does the Nsight Systems timeline show the warm-up separately from measured launches?
- Which time belongs to the kernel, and which time belongs to host-side orchestration?
- How does the effective bandwidth change when the number of elements increases?
