/*/
 *
 *  This is a program to test the performance
 *  of two different kernels, add_v1 and add_v2.
 *  
 *  The difference between the two kernels is that
 *  in add_v1, i is the fastest changing thread dimension,
 *  while in add_v2, j is the fastest changing thread dimension.
 * 
 *  Compile with:
 *      nvcc array_add_on_device.cu
 * 
 *  Run with:
 *     ./a.out
 * 
 *  Check the memory behavior of the two kernels with:
 *      ncu --set full ./array_add_on_device 4096 4096 16 16
 * 
/*/

#include <cstdio>
#include <cstdlib>

// Row-major flattening: consecutive col (j) values map to consecutive
// addresses (stride 1), while consecutive row (i) values are LDA elements
// apart (stride LDA). Whichever logical index we tie to the fastest-varying
// thread dimension therefore controls whether a warp's accesses land on
// contiguous addresses or on addresses LDA apart.
#define IDX(row, col, LDA) ((row) * (LDA) + (col))

// computes c(i,j) = a(i,j) + b(i,j)
// In this case i is the fastest changing thread dimension
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
        int idx = IDX(i, j, M);
        c[idx] = a[idx] + b[idx];
    }
}

// computes c(i,j) = a(i,j) + b(i,j)
// In this case j is the fastest changing thread dimension
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
    add_v1<<<blocks, threads>>>(a, b, c, n, m);
    add_v2<<<blocks, threads>>>(a, b, c, n, m);

    // Wait for both kernels to finish *before* freeing the memory they use.
    // cudaFree used to be called first here, which could release a, b, c
    // while the kernels were still (potentially) running against them.
    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
