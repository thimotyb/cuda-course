# Asynchronous Copy and Stream Overlap

This CUDA C++ example implements the staged copy-and-execute pattern from
M6.1.4. It uses pinned host memory, non-default streams, and independent data
chunks.

The program measures three views of the workload:

- copy only: host-to-device and device-to-host transfers;
- kernel only: arithmetic after the input is already on the GPU;
- full pipeline: host-to-device copy, kernel, and device-to-host copy.

Each view is measured with one stream and with multiple staged streams. This
isolates whether a speedup comes from overlapping transfers, overlapping kernel
work, or the complete copy-and-compute pipeline.

## Compile and run

From the repository root:

~~~bash
cd examples/ch06/async_overlap
nvcc async_overlap.cu -O3 -lineinfo -o async_overlap
./async_overlap
~~~

Arguments are:

~~~text
./async_overlap [chunks] [elements_per_chunk] [streams] [kernel_repetitions] [iterations]
~~~

The defaults use 8 chunks, 1,048,576 float elements per chunk, 4 streams,
100 arithmetic repetitions, and 5 measured iterations.

For a shorter first run:

~~~bash
./async_overlap 8 262144 4 50 3
~~~

The host buffer is allocated with cudaHostAlloc(). The program also reports
cudaDeviceProp.asyncEngineCount, which indicates the number of asynchronous
copy engines. A capability does not guarantee a speedup: the workload must be
large enough and the timeline must show concurrent work.

## Interpreting the output

The values in parentheses are one-stream time divided by staged time. For
example, 1.50x means that the staged version took approximately two thirds of
the one-stream time.

- A lower staged copy-only time indicates overlapping transfers.
- A lower staged kernel-only time indicates that independent kernel launches
  overlapped, although kernels may compete for compute resources.
- A lower full-pipeline staged time is the application-level result.
- Little or no improvement may indicate a transfer-bound workload, a
  compute-bound workload, insufficient chunk size, limited copy engines, or
  resource contention.

## When staged execution is not faster

A result below 1.00x means that the staged version is slower than the
one-stream baseline. This is an expected and useful result for some workloads.
For example:

~~~text
Copy only, one stream:     5.040 ms
Copy only, staged:         5.110 ms (0.99x)
Kernel only, one stream:   0.228 ms
Kernel only, staged:       0.192 ms (1.19x)
Full pipeline, one stream: 5.668 ms
Full pipeline, staged:     5.756 ms (0.98x)
~~~

This workload is transfer-bound: the copies take about 5 ms, while the kernel
takes only about 0.2 ms. The small kernel improvement cannot compensate for
the extra stream management and synchronization overhead. If the device
reports asyncEngineCount: 1, it has one asynchronous copy engine, so
host-to-device and device-to-host transfers cannot both occupy separate copy
engines at the same time. The kernel can still overlap with transfers, but
there is very little computation available to hide.

To make the overlap easier to observe, increase the arithmetic work:

~~~bash
./async_overlap 8 1048576 4 2000 5
~~~

When copy and kernel times are comparable, the staged pipeline may approach the
larger of the two times per chunk rather than their sum. More streams do not
guarantee a speedup: they add scheduling and synchronization work and can
compete for the same memory-bandwidth or compute resources.

When transfers dominate, the better optimization may be to transfer less data,
keep intermediate results on the GPU, batch small transfers, or increase the
amount of useful computation performed for each transfer. The purpose of this
exercise is therefore also to identify when staged execution is not the right
first optimization.

## Long-kernel variant

The default kernel may be too short to make the overlap visible. Keep the
default run as the lightweight baseline, then run a second version with more
arithmetic repetitions:

~~~bash
./async_overlap 8 1048576 4 2000 5
~~~

The fourth argument is kernel_repetitions. Increasing it deliberately makes
the kernel spend more time computing, so transfers for other chunks have a
larger opportunity to overlap with computation. The goal is not to claim that
the extra arithmetic is useful application work; it is a controlled teaching
variant that makes the scheduling pattern visible in Nsight Systems.

Compare the full-pipeline results from both runs. With the short kernel, the
copy time can dominate and staged execution may be equal or slower. With the
long kernel, the total staged time may improve if the GPU can execute the
kernel while its copy engine transfers another chunk. The exact result depends
on chunk size, stream count, copy-engine count, memory bandwidth, and compute
resource contention.

## Nsight Systems

Capture a timeline for a short run:

~~~bash
nsys profile --trace=cuda --stats=true --force-overwrite=true \
  -o async-overlap-nsys ./async_overlap 8 262144 4 50 3
nsys-ui async-overlap-nsys.nsys-rep
~~~

Inspect the CUDA API and GPU hardware rows. In the staged full-pipeline phase,
look for host-to-device copies, transform kernels, and device-to-host copies
occupying overlapping intervals. Compare that timeline with the program's
full-pipeline timings.

## Questions

- Does staged copy-only improve, and does asyncEngineCount explain the
  timeline?
- Does staged kernel-only improve, or are the kernels already saturating the
  compute resources?
- Which phase limits the full pipeline: transfers, kernel execution, or
  launch and synchronization overhead?
- What changes when chunk size or stream count changes?
- Does the staged version achieve real overlap, supported by both timings and
  the Nsight Systems timeline?

## Cleanup

~~~bash
rm -f async_overlap async-overlap-nsys.nsys-rep async-overlap-nsys.sqlite
~~~
