/*/
 *
 *   A simple program to demonstrate array indexing in kernels.
 *
 *  Compile with:
 *      nvcc array_assign_on_device.cu
 *
 *  Run with:
 *     ./a.out
 *
/*/

// This demo's whole point is the indexing expression inside the kernel:
// each thread computes its own global index from its block/thread coordinates
// and writes exactly one element of the array. It is intentionally the
// smallest possible example of the "one thread, one element" pattern that
// almost every CUDA kernel builds on, so memory management is kept out of
// the way (see the cudaMallocManaged note below) to keep the focus there.

#include <stdio.h>

// Step 1: each thread computes its own flat index i from the block/thread
// coordinates the launch configuration gives it, and writes to a[i]. With
// blocks * threads >= N, every array element gets written by exactly one
// thread, in parallel, with no two threads touching the same index.
__global__ void kernel(int *a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = i;
}

int main()
{

    // Step 2: pick a problem size and a launch configuration. 128 threads per
    // block is an arbitrary but reasonable block size; blocks is rounded up
    // so that blocks * threads covers all N elements (ceiling division).
    int N = 4096;
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    int *a;

    // Step 3: allocate with cudaMallocManaged rather than cudaMalloc. This
    // gives one pointer, `a`, that is valid from both host and device code:
    // the CUDA runtime automatically migrates the underlying pages between
    // CPU and GPU memory as each side touches them, so there is no separate
    // host buffer and no explicit cudaMemcpy to write. That keeps this
    // example focused purely on the indexing pattern above, at the cost of
    // migration overhead that a performance-sensitive program would instead
    // avoid with explicit cudaMalloc + cudaMemcpy (see the ch05 bandwidth
    // examples for that approach).
    cudaMallocManaged(&a, N * sizeof(int));

    // Step 4: launch the kernel so every element of `a` is written on the GPU.
    kernel<<<blocks, threads>>>(a, N);

    // Step 5: wait for the kernel to finish. This is required before the CPU
    // reads `a` below — with managed memory it is what makes it safe for the
    // runtime to migrate the pages back to the host on first host access.
    cudaDeviceSynchronize();

    // Step 6: read the first 10 elements from the CPU, through the very same
    // pointer that the kernel just wrote on the GPU.
    for (int i = 0; i < 10; i++)
        printf("%d\n", a[i]);

    // Step 7: release the managed allocation.
    cudaFree(a);
    return 0;
}
