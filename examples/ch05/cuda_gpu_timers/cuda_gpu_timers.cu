#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t status, const char *operation) {
  // Report the exact CUDA API call that failed; timing results are meaningless
  // if the workload was not launched or completed correctly.
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// Each thread computes one independent output element. This gives the example
// enough parallel work while keeping the operation easy to validate.
__global__ void addVectors(const float *a, const float *b, float *c,
                           int count) {
  // The grid can be rounded up, so the final block may contain out-of-range
  // threads. The guard prevents those threads from writing past c.
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

static double millisecondsSince(
    const std::chrono::steady_clock::time_point &start,
    const std::chrono::steady_clock::time_point &stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

int main(int argc, char **argv) {
  int count = argc > 1 ? std::atoi(argv[1]) : 1 << 24;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 10;
  if (count <= 0 || iterations <= 0) {
    std::fprintf(stderr, "count and iterations must be positive\n");
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp properties{};
  checkCuda(cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
  checkCuda(cudaSetDevice(device), "cudaSetDevice");

  // Allocate and initialize data before timing. Allocation and host-device
  // transfers are separate costs from the kernel interval measured below.
  size_t bytes = static_cast<size_t>(count) * sizeof(float);
  std::vector<float> host_a(count, 1.0f);
  std::vector<float> host_b(count, 2.0f);
  std::vector<float> host_c(count, 0.0f);
  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  checkCuda(cudaMalloc(&device_a, bytes), "cudaMalloc(device_a)");
  checkCuda(cudaMalloc(&device_b, bytes), "cudaMalloc(device_b)");
  checkCuda(cudaMalloc(&device_c, bytes), "cudaMalloc(device_c)");
  checkCuda(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(a)");
  checkCuda(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(b)");

  // Use a conventional block size for this bandwidth-oriented kernel.
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  // Warm up the CUDA context and kernel path. The first launch can include
  // one-time runtime overhead that should not dominate steady-state timing.
  addVectors<<<blocks, threads>>>(device_a, device_b, device_c, count);
  checkCuda(cudaGetLastError(), "warm-up kernel launch");
  checkCuda(cudaDeviceSynchronize(), "warm-up synchronization");

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  checkCuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

  double cpu_total_ms = 0.0;
  float event_total_ms = 0.0f;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    // The two measurements use separate launches. This keeps each timing
    // method easy to read, but the reported averages should be compared as
    // measurement methods, not as one shared execution of the kernel.
    // A CPU timer needs synchronization because the kernel launch is asynchronous.
    checkCuda(cudaDeviceSynchronize(), "CPU timer start synchronization");
    auto cpu_start = std::chrono::steady_clock::now();
    addVectors<<<blocks, threads>>>(device_a, device_b, device_c, count);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "CPU timer stop synchronization");
    auto cpu_stop = std::chrono::steady_clock::now();
    cpu_total_ms += millisecondsSince(cpu_start, cpu_stop);

    // Events are placed in the same stream as the kernel and timestamped by the
    // GPU. The CPU does not need to measure the interval with its own clock.
    checkCuda(cudaEventRecord(start_event, 0), "cudaEventRecord(start)");
    addVectors<<<blocks, threads>>>(device_a, device_b, device_c, count);
    checkCuda(cudaGetLastError(), "event-timed kernel launch");
    checkCuda(cudaEventRecord(stop_event, 0), "cudaEventRecord(stop)");
    // Waiting for the stop event makes elapsed time available without requiring
    // a broader device-wide synchronization for this particular interval.
    checkCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    float event_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&event_ms, start_event, stop_event),
              "cudaEventElapsedTime");
    event_total_ms += event_ms;
  }

  // Copy one completed result back to the host so the example checks correctness
  // as well as timing. Validation is not included in either timed interval.
  checkCuda(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(result)");
  if (host_c[0] != 3.0f) {
    std::fprintf(stderr, "validation failed: expected 3.0, got %f\n", host_c[0]);
    return EXIT_FAILURE;
  }

  std::printf("GPU: %s\n", properties.name);
  std::printf("Elements: %d\n", count);
  std::printf("Measured iterations: %d\n", iterations);
  std::printf("CPU wall-clock time: %.3f ms\n", cpu_total_ms / iterations);
  std::printf("CUDA event time: %.3f ms\n", event_total_ms / iterations);
  std::printf("CPU/event ratio: %.2fx\n",
              (cpu_total_ms / iterations) / (event_total_ms / iterations));
  std::printf("Validation: OK\n");

  checkCuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
  checkCuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
  checkCuda(cudaFree(device_a), "cudaFree(device_a)");
  checkCuda(cudaFree(device_b), "cudaFree(device_b)");
  checkCuda(cudaFree(device_c), "cudaFree(device_c)");
  return EXIT_SUCCESS;
}
