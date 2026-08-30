#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void checkCuda(cudaError_t status, const char *operation) {
  // Stop at the failing CUDA call. A timing result is useful only when the
  // kernel launch and all device operations completed successfully.
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// Adjacent threads read adjacent elements and write adjacent elements.
// This is the access pattern expected to make global-memory transactions
// efficient because requests from one warp can be coalesced.
__global__ void coalescedCopy(const float *input, float *output, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index];
  }
}

// The logical work is the same as coalescedCopy, but consecutive threads are
// separated by stride elements. The gaps force the hardware to fetch more
// memory transactions than the kernel's useful bytes require.
__global__ void stridedCopy(const float *input, float *output, int count,
                            int stride) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    int position = index * stride;
    output[position] = input[position];
  }
}

static float timeCoalesced(const float *input, float *output, int count,
                           int blocks, int threads, int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  float total_ms = 0.0f;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    // Events measure the GPU timeline. Both events use the default stream,
    // which is also where this kernel is launched.
    checkCuda(cudaEventRecord(start, 0), "cudaEventRecord(start)");
    coalescedCopy<<<blocks, threads>>>(input, output, count);
    checkCuda(cudaGetLastError(), "coalescedCopy launch");
    checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
              "cudaEventElapsedTime");
    total_ms += elapsed_ms;
  }

  checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  return total_ms / iterations;
}

static float timeStrided(const float *input, float *output, int count,
                         int stride, int blocks, int threads, int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  float total_ms = 0.0f;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    checkCuda(cudaEventRecord(start, 0), "cudaEventRecord(start)");
    stridedCopy<<<blocks, threads>>>(input, output, count, stride);
    checkCuda(cudaGetLastError(), "stridedCopy launch");
    checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
              "cudaEventElapsedTime");
    total_ms += elapsed_ms;
  }

  checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  return total_ms / iterations;
}

static double effectiveBandwidthGBs(int count, float elapsed_ms) {
  // A copy reads one float and writes one float: two transfers of 4 bytes.
  // Use decimal GB here, matching the convention in the course explanation.
  double useful_bytes = static_cast<double>(count) * sizeof(float) * 2.0;
  double elapsed_seconds = elapsed_ms / 1000.0;
  return (useful_bytes / 1.0e9) / elapsed_seconds;
}

int main(int argc, char **argv) {
  int count = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
  int stride = argc > 2 ? std::atoi(argv[2]) : 16;
  int iterations = argc > 3 ? std::atoi(argv[3]) : 10;
  if (count <= 0 || stride <= 0 || iterations <= 0) {
    std::fprintf(stderr, "count, stride, and iterations must be positive\n");
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp properties{};
  checkCuda(cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
  checkCuda(cudaSetDevice(device), "cudaSetDevice");

  // The strided kernel touches count * stride elements, so allocate enough
  // storage for its final position. The useful traffic is still count elements.
  size_t allocation_count = static_cast<size_t>(count) * stride;
  size_t bytes = allocation_count * sizeof(float);
  float *input = nullptr;
  float *output = nullptr;
  checkCuda(cudaMalloc(&input, bytes), "cudaMalloc(input)");
  checkCuda(cudaMalloc(&output, bytes), "cudaMalloc(output)");
  checkCuda(cudaMemset(input, 1, bytes), "cudaMemset(input)");

  int threads = 256;
  int blocks = (count + threads - 1) / threads;

  // Warm up both code paths so context initialization is not reported as
  // memory bandwidth. The synchronization also guarantees a clean start.
  coalescedCopy<<<blocks, threads>>>(input, output, count);
  stridedCopy<<<blocks, threads>>>(input, output, count, stride);
  checkCuda(cudaGetLastError(), "warm-up launch");
  checkCuda(cudaDeviceSynchronize(), "warm-up synchronization");

  float coalesced_ms =
      timeCoalesced(input, output, count, blocks, threads, iterations);
  float strided_ms = timeStrided(input, output, count, stride, blocks, threads,
                                 iterations);

  std::printf("GPU: %s\n", properties.name);
  std::printf("Useful elements copied: %d\n", count);
  std::printf("Stride case: %d\n", stride);
  std::printf("Iterations: %d\n", iterations);
  std::printf("Coalesced kernel time: %.3f ms\n", coalesced_ms);
  std::printf("Coalesced effective bandwidth: %.2f GB/s\n",
              effectiveBandwidthGBs(count, coalesced_ms));
  std::printf("Strided kernel time: %.3f ms\n", strided_ms);
  std::printf("Strided effective bandwidth: %.2f GB/s\n",
              effectiveBandwidthGBs(count, strided_ms));
  std::printf("Bandwidth retained by strided case: %.2f%%\n",
              100.0 * effectiveBandwidthGBs(count, strided_ms) /
                  effectiveBandwidthGBs(count, coalesced_ms));
  std::printf("Theoretical peak bandwidth must be read from the GPU specifications.\n");

  checkCuda(cudaFree(input), "cudaFree(input)");
  checkCuda(cudaFree(output), "cudaFree(output)");
  return EXIT_SUCCESS;
}

