#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

static void checkCuda(cudaError_t result, const char *message) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(result));
    exit(1);
  }
}

int main() {
  int device_count = 0;
  checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");

  if (device_count == 0) {
    printf("No CUDA-capable device found.\n");
    return 0;
  }

  printf("CUDA devices found: %d\n\n", device_count);

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, device),
              "cudaGetDeviceProperties failed");

    printf("Device %d\n", device);
    printf("  name: %s\n", prop.name);
    printf("  compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("  registers per block: %d\n", prop.regsPerBlock);
    printf("  registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  global memory: %.2f GiB\n",
           static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    printf("\n");
  }

  return 0;
}
