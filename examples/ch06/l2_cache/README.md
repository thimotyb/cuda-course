# L2 Cache Access Policy

This CUDA C++ example compares repeated reads of the same global-memory
working set with three L2 access-policy hints:

- normal access, without a stream policy window;
- streaming access, which tells the cache that the data is not expected to be
  reused;
- persisting access, which tells the cache that the data is expected to be
  reused.

The hints are not strict commands. The result depends on the GPU, the working
set size, the available L2 set-aside capacity, and other activity on the
device. The exercise is therefore a timing experiment, not a promise that
persisting access is always faster.

## Compile and run

~~~bash
cd examples/ch06/l2_cache
nvcc l2_cache.cu -O3 -lineinfo -o l2_cache
./l2_cache
~~~

Arguments are:

~~~text
./l2_cache [elements] [repetitions] [iterations]
~~~

The default working set contains 1,048,576 floats, about 4 MiB, and is read
repeatedly by the kernel. Try a small working set first:

~~~bash
./l2_cache 262144 2000 5
~~~

Then compare it with a working set that is larger than the available L2 cache:

~~~bash
./l2_cache 67108864 20 5
~~~

The program reports the device L2 size, the maximum L2 set-aside size, and
whether the device supports the access-policy window. If the device does not
support the required property, the persisting comparison is skipped.

## Interpreting the output

The normal case is a baseline. The streaming case is a hint that repeated
reuse is not expected. The persisting case is a hint that the working set
should remain in the L2 set-aside region when possible.

- A persisting result faster than streaming suggests that L2 reuse helped.
- Similar results indicate that the working set, kernel, or device did not
  expose a measurable cache difference.
- A persisting result slower than normal or streaming may indicate that the
  working set exceeds the set-aside capacity and causes cache thrashing.
- A small working set can already be served efficiently by ordinary caching,
  leaving little room for an additional persisting hint.

The time includes the repeated kernel reads and is measured with CUDA events on
the GPU stream. The program also prints a checksum so the compiler cannot
remove the memory reads as dead work.

## Nsight Compute

Profile one short run:

~~~bash
ncu --section MemoryWorkloadAnalysis --section SpeedOfLight \
  --kernel-name regex:'^readWorkingSet' \
  --launch-skip 1 --launch-count 3 --force-overwrite \
  -o l2-cache-ncu ./l2_cache 262144 2000 5
ncu-ui l2-cache-ncu.ncu-rep
~~~

Inspect L2 hit rate, DRAM traffic, achieved memory throughput, and kernel
duration. Compare those metrics with the timings printed by the program.

## Questions

- Does the persisting hint improve the small working set?
- What changes when the working set becomes larger than L2?
- Does the streaming hint reduce L2 reuse or only change the measured time
  slightly?
- Is the persisting region small enough for the reported set-aside capacity?
- Do the profiler counters support the timing result?

## Cleanup

~~~bash
rm -f l2_cache l2-cache-ncu.ncu-rep l2-cache-ncu.sqlite
~~~
