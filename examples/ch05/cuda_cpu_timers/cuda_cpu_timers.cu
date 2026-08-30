#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

static void checkCuda(cudaError_t status, const char *operation) {
  // CUDA calls return an error code. Stop at the operation that failed so the
  // timing example does not continue with an invalid device state.
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

__global__ void addVectors(const float *a, const float *b, float *c,
                           int count) {
  // The grid-stride mapping is simplified here to one thread per element.
  // The bounds check is needed because the rounded-up grid may contain extra
  // threads when count is not an exact multiple of the block size.
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

int main(int argc, char **argv) {
  int count = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
  if (count <= 0) {
    std::fprintf(stderr, "count must be positive\n");
    return EXIT_FAILURE;
  }

  // cudaMalloc allocates device memory. Allocation is intentionally outside
  // the timed interval because this example focuses on kernel timing.
  size_t bytes = static_cast<size_t>(count) * sizeof(float);
  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;
  checkCuda(cudaMalloc(&a, bytes), "cudaMalloc(a)");
  checkCuda(cudaMalloc(&b, bytes), "cudaMalloc(b)");
  checkCuda(cudaMalloc(&c, bytes), "cudaMalloc(c)");

  // A block size of 256 is a common starting point for an element-wise kernel.
  // It is a launch configuration choice, not part of the timer mechanism.
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  addVectors<<<blocks, threads>>>(a, b, c, count);
  checkCuda(cudaGetLastError(), "warm-up launch");
  checkCuda(cudaDeviceSynchronize(), "warm-up synchronization");

  // This warm-up removes one-time initialization effects from the measurement.
  // CUDA launches are asynchronous, so wait before starting the host timer.
  checkCuda(cudaDeviceSynchronize(), "timer start synchronization");
  // The CPU clock starts after the device is idle. The launch itself returns
  // quickly, but the following synchronization keeps the full GPU execution
  // inside the measured host interval.
  auto start = std::chrono::steady_clock::now();
  addVectors<<<blocks, threads>>>(a, b, c, count);
  checkCuda(cudaGetLastError(), "timed launch");

  // Without this wait, the CPU timer would stop after submission, not completion.
  checkCuda(cudaDeviceSynchronize(), "timer stop synchronization");
  auto stop = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(stop - start).count();

  // A CPU timer measures what the host observes: launch overhead plus the wait
  // for completion, rather than only the instructions executed by the GPU.
  std::printf("Elements: %d\n", count);
  std::printf("CPU wall-clock interval: %.3f ms\n", elapsed_ms);
  std::printf("The interval includes host launch and synchronization overhead.\n");

  checkCuda(cudaFree(a), "cudaFree(a)");
  checkCuda(cudaFree(b), "cudaFree(b)");
  checkCuda(cudaFree(c), "cudaFree(c)");
  return EXIT_SUCCESS;
}
