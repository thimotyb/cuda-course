#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void checkCuda(cudaError_t status, const char *operation) {
  // Stop at the failing CUDA call so that no invalid timing is reported.
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// Every thread moves one float from input to output. The kernel is intentionally
// simple: M5 focuses on measuring memory movement, not yet on access patterns.
__global__ void copyKernel(const float *input, float *output, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index];
  }
}

int main(int argc, char **argv) {
  int count = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  if (count <= 0 || iterations <= 0) {
    std::fprintf(stderr, "count and iterations must be positive\n");
    return EXIT_FAILURE;
  }

  size_t bytes = static_cast<size_t>(count) * sizeof(float);
  float *device_input = nullptr;
  float *device_output = nullptr;
  checkCuda(cudaMalloc(&device_input, bytes), "cudaMalloc(input)");
  checkCuda(cudaMalloc(&device_output, bytes), "cudaMalloc(output)");
  checkCuda(cudaMemset(device_input, 1, bytes), "cudaMemset(input)");

  int threads = 256;
  int blocks = (count + threads - 1) / threads;

  // The first launch may initialize the CUDA context. Warm up before measuring.
  copyKernel<<<blocks, threads>>>(device_input, device_output, count);
  checkCuda(cudaGetLastError(), "warm-up launch");
  checkCuda(cudaDeviceSynchronize(), "warm-up synchronization");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  float total_ms = 0.0f;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    // CUDA events timestamp the GPU stream. The stop event is synchronized
    // before its elapsed time is read.
    checkCuda(cudaEventRecord(start, 0), "cudaEventRecord(start)");
    copyKernel<<<blocks, threads>>>(device_input, device_output, count);
    checkCuda(cudaGetLastError(), "copyKernel launch");
    checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
              "cudaEventElapsedTime");
    total_ms += elapsed_ms;
  }

  // A copy reads one float and writes one float, so useful traffic is 8 bytes
  // per element. This is a logical bandwidth estimate, not a hardware peak.
  double average_ms = total_ms / iterations;
  double useful_bytes = static_cast<double>(count) * sizeof(float) * 2.0;
  double effective_gb_s = (useful_bytes / 1.0e9) / (average_ms / 1000.0);

  std::printf("Useful elements copied: %d\n", count);
  std::printf("Measured iterations: %d\n", iterations);
  std::printf("Average kernel time: %.3f ms\n", average_ms);
  std::printf("Effective bandwidth: %.2f GB/s\n", effective_gb_s);
  std::printf("Useful traffic per iteration: %.2f MB\n",
              useful_bytes / 1.0e6);

  checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  checkCuda(cudaFree(device_input), "cudaFree(input)");
  checkCuda(cudaFree(device_output), "cudaFree(output)");
  return EXIT_SUCCESS;
}

