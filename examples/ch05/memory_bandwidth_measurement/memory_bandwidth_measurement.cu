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
  // Step 1: parse element count and iteration count from argv, with defaults.
  // argv[1] = count: number of floats to copy input->output, i.e. the size of
  //   the input/output buffers (count * sizeof(float) bytes each). Defaults
  //   to 1 << 24 (16,777,216 elements, 64 MB per buffer).
  // argv[2] = iterations: how many timed kernel launches to average over.
  //   Defaults to 20.
  // Usage: ./memory_bandwidth_measurement [count] [iterations]
  //   e.g.  ./memory_bandwidth_measurement 1048576 100
  int count = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  if (count <= 0 || iterations <= 0) {
    std::fprintf(stderr, "count and iterations must be positive\n");
    return EXIT_FAILURE;
  }

  // Step 2: allocate the input/output device buffers and seed the input.
  size_t bytes = static_cast<size_t>(count) * sizeof(float);
  float *device_input = nullptr;
  float *device_output = nullptr;
  checkCuda(cudaMalloc(&device_input, bytes), "cudaMalloc(input)");
  checkCuda(cudaMalloc(&device_output, bytes), "cudaMalloc(output)");
  checkCuda(cudaMemset(device_input, 1, bytes), "cudaMemset(input)");

  int threads = 256;
  int blocks = (count + threads - 1) / threads;

  // Step 3: warm-up launch. The first launch may initialize the CUDA
  // context, so it is run and synchronized once before any timing starts.
  copyKernel<<<blocks, threads>>>(device_input, device_output, count);
  checkCuda(cudaGetLastError(), "warm-up launch");
  checkCuda(cudaDeviceSynchronize(), "warm-up synchronization");

  // Step 4: create the CUDA events used to time each kernel launch.
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  float total_ms = 0.0f;

  // Step 5: repeatedly launch the copy kernel, timing each launch with CUDA
  // events. Events timestamp the GPU stream directly, and the stop event is
  // synchronized before its elapsed time is read, so the result reflects the
  // kernel's actual device time.
  for (int iteration = 0; iteration < iterations; ++iteration) {
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

  // Step 6: derive the effective bandwidth from the average kernel time. A
  // copy reads one float and writes one float, so useful traffic is 8 bytes
  // per element. This is a logical bandwidth estimate, not a hardware peak.
  double average_ms = total_ms / iterations;
  double useful_bytes = static_cast<double>(count) * sizeof(float) * 2.0;
  double effective_gb_s = (useful_bytes / 1.0e9) / (average_ms / 1000.0);

  // Calculate the device's theoretical DRAM bandwidth from runtime properties.
  // memoryClockRate is reported in kHz and memoryBusWidth in bits. The factor
  // of two accounts for double-data-rate memory transfers.
  cudaDeviceProp properties{};
  checkCuda(cudaGetDeviceProperties(&properties, 0),
            "cudaGetDeviceProperties");
  double theoretical_gb_s =
      (static_cast<double>(properties.memoryClockRate) * 1000.0 *
       (static_cast<double>(properties.memoryBusWidth) / 8.0) * 2.0) /
      1.0e9;

  // Step 7: report the results.
  std::printf("GPU: %s\n", properties.name);
  std::printf("Memory clock: %.0f MHz\n",
              properties.memoryClockRate / 1000.0);
  std::printf("Memory bus width: %d bits\n", properties.memoryBusWidth);
  std::printf("Theoretical peak bandwidth: %.2f GB/s\n", theoretical_gb_s);
  std::printf("Useful elements copied: %d\n", count);
  std::printf("Measured iterations: %d\n", iterations);
  std::printf("Average kernel time: %.3f ms\n", average_ms);
  std::printf("Effective bandwidth: %.2f GB/s\n", effective_gb_s);
  std::printf("Effective/theoretical bandwidth: %.2f%%\n",
              100.0 * effective_gb_s / theoretical_gb_s);
  std::printf("Useful traffic per iteration: %.2f MB\n",
              useful_bytes / 1.0e6);

  // Step 8: release CUDA events and device buffers before exiting.
  checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  checkCuda(cudaFree(device_input), "cudaFree(input)");
  checkCuda(cudaFree(device_output), "cudaFree(output)");
  return EXIT_SUCCESS;
}
