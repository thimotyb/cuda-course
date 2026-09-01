/*/
 *
 *  This is a program to test the performance
 *  of two different kernels, add_v1 and add_v2.
 *  
 *  The difference between the two kernels is that
 *  in add_v1, i is the fastest changing thread dimension,
 *  while in add_v2, j is the fastest changing thread dimension.
 *
 *  This matters because the array is stored row-major: idx = i * M + j, so
 *  consecutive j values are 1 apart in memory but consecutive i values are
 *  M apart. And it is threadIdx.x, not threadIdx.y, that actually varies
 *  fastest across the 32 threads of a warp. So whichever index (i or j) is
 *  tied to threadIdx.x decides whether a warp's memory accesses are
 *  contiguous or scattered:
 *
 *  Why is it always threadIdx.x that varies fastest? Not because x is
 *  special -- it is a fixed convention of the CUDA programming model for how
 *  a (possibly 3D) thread block is linearized into the single ordering used
 *  to form warps:
 *      linear_id = threadIdx.z * (blockDim.y * blockDim.x)
 *                + threadIdx.y * blockDim.x
 *                + threadIdx.x
 *  This is a mixed-radix number where threadIdx.x is the least-significant
 *  "digit" -- exactly like the units digit changes on every count while the
 *  tens digit only changes every ten counts. A warp is just 32 consecutive
 *  threads in this linear_id order, so within a warp threadIdx.x cycles
 *  through its whole range before threadIdx.y ever increments. The
 *  programmer's job is simply to line up the contiguous (stride-1) dimension
 *  of their data with threadIdx.x, which is exactly what add_v2 does below.
 *    - add_v1 ties i to threadIdx.x -> consecutive threads jump by M in
 *      memory -> STRIDED / uncoalesced accesses (slow: one memory
 *      transaction per thread instead of one per warp).
 *    - add_v2 ties j to threadIdx.x -> consecutive threads land on
 *      consecutive addresses -> COALESCED accesses (fast: the GPU merges
 *      the whole warp's accesses into one wide transaction).
 *  Same math, same result -- only the memory access pattern differs.
 *
 *  Compile with:
 *      nvcc array_add_on_device.cu -o array_add_on_device
 *
 *  Run with:
 *     ./array_add_on_device <m> <n> <t_y> <t_x>
 *     e.g. ./array_add_on_device 4096 4096 16 16
 *
 *  where:
 *     m   = number of columns of the (conceptual) n x m matrix a/b/c
 *           represent -- it is the row stride used by IDX (M above).
 *     n   = number of rows of that matrix.
 *     t_y = thread block dimension along y (threads.y).
 *     t_x = thread block dimension along x (threads.x).
 *  a, b, c hold m * n elements total, and the 2D grid of blocks/threads is
 *  sized as blocks(n / t_x, m / t_y) so that one thread covers exactly one
 *  matrix element -- the whole array is processed in one parallel pass, not
 *  a sequential loop. Note: n must be divisible by t_x and m by t_y (integer
 *  division), and t_x * t_y must not exceed 1024 (max threads per block).
 *
 *  Check the memory behavior of the two kernels with:
 *      ncu --set full ./array_add_on_device 4096 4096 16 16
 *
/*/

#include <cstdio>
#include <cstdlib>

// Row-major flattening: consecutive col (j) values map to consecutive
// addresses (stride 1), while consecutive row (i) values are M elements
// apart (stride M, the row length). Whichever logical index we tie to the
// fastest-varying thread dimension therefore controls whether a warp's
// accesses land on contiguous addresses or on addresses M apart.
#define IDX(row, col, M) ((row) * (M) + (col))

// computes c(i,j) = a(i,j) + b(i,j)
// In this case i is the fastest changing thread dimension (because is linked to threadIdx.x)
//
// Within a warp, threadIdx.x is what actually varies fastest across the 32
// consecutive threads (threadIdx.y stays fixed for many of them). Here i is
// derived from threadIdx.x, so consecutive threads in a warp get consecutive
// i values with the *same* j. Since idx = i * M + j, that means consecutive
// threads compute idx values M apart -- a strided access pattern. The GPU
// cannot merge these into one wide memory transaction, so each thread's
// load/store to a/b/c effectively needs its own transaction: uncoalesced,
// memory-bandwidth-inefficient access.
__global__ void add_v1(int *a, int *b, int *c, int N, int M)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < M)
    {
        // Flatten the 2D (i, j) position into the real 1D index used to
        // address a/b/c, which are plain linear buffers in CUDA memory.
        int idx = IDX(i, j, M);
        c[idx] = a[idx] + b[idx];
    }
}

// computes c(i,j) = a(i,j) + b(i,j)
// In this case j is the fastest changing thread dimension (because is linked to threadIdx.x)
//
// Here j (the column, the contiguous dimension per IDX above) is derived
// from threadIdx.x instead. Consecutive threads in a warp now get
// consecutive j values with the same i, so idx = i * M + j increases by 1
// per thread -- consecutive addresses. The GPU can coalesce these into a
// single wide memory transaction per warp, so this version does the exact
// same math as add_v1 but with far more efficient global memory traffic.
__global__ void add_v2(int *a, int *b, int *c, int N, int M)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < M)
    {
        // Flatten the 2D (i, j) position into the real 1D index used to
        // address a/b/c, which are plain linear buffers in CUDA memory.
        int idx = IDX(i, j, M);
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char *argv[])
{

    int m, n, t_y, t_x;
    int *a, *b, *c;

    if(argc != 5){
        printf("Usage: %s <m> <n> <t_y> <t_x>\n", argv[0]);
        return 1;
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    t_y = atoi(argv[3]);
    t_x = atoi(argv[4]);

    cudaMallocManaged(&a, m * n * sizeof(int));
    cudaMallocManaged(&b, m * n * sizeof(int));
    cudaMallocManaged(&c, m * n * sizeof(int));

    dim3 threads(t_x, t_y);
    dim3 blocks(n / threads.x, m / threads.y);

    // CUDA events time GPU work directly on the device's own clock, so they
    // capture each kernel's real execution time -- not host-side overhead
    // like the launch call itself. Both kernels run on the default stream,
    // which is implicitly synchronizing, so add_v2 only starts once add_v1
    // has actually finished: each pair of events below brackets exactly one
    // kernel's execution.
    cudaEvent_t start_v1, stop_v1, start_v2, stop_v2;
    cudaEventCreate(&start_v1);
    cudaEventCreate(&stop_v1);
    cudaEventCreate(&start_v2);
    cudaEventCreate(&stop_v2);

    cudaEventRecord(start_v1);
    add_v1<<<blocks, threads>>>(a, b, c, n, m);
    cudaEventRecord(stop_v1);

    cudaEventRecord(start_v2);
    add_v2<<<blocks, threads>>>(a, b, c, n, m);
    cudaEventRecord(stop_v2);

    // Wait for both kernels to finish *before* freeing the memory they use.
    // cudaFree used to be called first here, which could release a, b, c
    // while the kernels were still (potentially) running against them.
    // This also guarantees the stop events above have completed, so their
    // recorded timestamps are ready to read below.
    cudaDeviceSynchronize();

    float ms_v1 = 0.0f, ms_v2 = 0.0f;
    cudaEventElapsedTime(&ms_v1, start_v1, stop_v1);
    cudaEventElapsedTime(&ms_v2, start_v2, stop_v2);
    printf("add_v1 (strided,   uncoalesced): %.3f ms\n", ms_v1);
    printf("add_v2 (contiguous, coalesced):  %.3f ms\n", ms_v2);
    printf("speedup of add_v2 over add_v1:   %.2fx\n", ms_v1 / ms_v2);

    cudaEventDestroy(start_v1);
    cudaEventDestroy(stop_v1);
    cudaEventDestroy(start_v2);
    cudaEventDestroy(stop_v2);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
