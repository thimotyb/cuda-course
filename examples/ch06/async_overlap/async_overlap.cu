#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// Independent arithmetic keeps the compute phase visible in Nsight Systems.
// Different chunks can therefore be submitted to different streams safely.
__global__ void transform(float *data, int count, int repetitions) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = data[index];
    for (int r = 0; r < repetitions; ++r) {
      value = value * 1.000001f + 0.000001f;
    }
    data[index] = value;
  }
}

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

static std::vector<cudaStream_t> makeStreams(int count) {
  std::vector<cudaStream_t> streams(count);
  for (cudaStream_t &stream : streams) {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
  }
  return streams;
}

static void sync(const std::vector<cudaStream_t> &streams) {
  for (cudaStream_t stream : streams) {
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
}

static void destroy(std::vector<cudaStream_t> &streams) {
  for (cudaStream_t stream : streams) {
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  }
}

// One non-default stream is the serialized baseline.
static double copyOneStream(float *host, float *device, size_t chunk_bytes,
                            int chunks, int iterations) {
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
  auto start = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < chunks; ++c) {
      char *h = reinterpret_cast<char *>(host) + c * chunk_bytes;
      char *d = reinterpret_cast<char *>(device) + c * chunk_bytes;
      check(cudaMemcpyAsync(d, h, chunk_bytes, cudaMemcpyHostToDevice, stream),
            "host-to-device copy");
      check(cudaMemcpyAsync(h, d, chunk_bytes, cudaMemcpyDeviceToHost, stream),
            "device-to-host copy");
    }
    check(cudaStreamSynchronize(stream), "copy synchronization");
  }
  double result = elapsed(start) / iterations;
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return result;
}

// Independent chunks use different streams and can overlap on capable GPUs.
static double copyStaged(float *host, float *device, size_t chunk_bytes,
                         int chunks, int stream_count, int iterations) {
  std::vector<cudaStream_t> streams = makeStreams(stream_count);
  auto start = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < chunks; ++c) {
      cudaStream_t stream = streams[c % stream_count];
      char *h = reinterpret_cast<char *>(host) + c * chunk_bytes;
      char *d = reinterpret_cast<char *>(device) + c * chunk_bytes;
      check(cudaMemcpyAsync(d, h, chunk_bytes, cudaMemcpyHostToDevice, stream),
            "staged host-to-device copy");
      check(cudaMemcpyAsync(h, d, chunk_bytes, cudaMemcpyDeviceToHost, stream),
            "staged device-to-host copy");
    }
    sync(streams);
  }
  double result = elapsed(start) / iterations;
  destroy(streams);
  return result;
}

static double kernelOneStream(float *device, int elements, int chunks,
                              int repetitions, int iterations, int blocks,
                              int threads) {
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
  auto start = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < chunks; ++c) {
      transform<<<blocks, threads, 0, stream>>>(
          device + c * elements, elements, repetitions);
      check(cudaGetLastError(), "transform launch");
    }
    check(cudaStreamSynchronize(stream), "kernel synchronization");
  }
  double result = elapsed(start) / iterations;
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return result;
}

static double kernelStaged(float *device, int elements, int chunks,
                           int stream_count, int repetitions, int iterations,
                           int blocks, int threads) {
  std::vector<cudaStream_t> streams = makeStreams(stream_count);
  auto start = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < chunks; ++c) {
      int stream_id = c % stream_count;
      transform<<<blocks, threads, 0, streams[stream_id]>>>(
          device + c * elements, elements, repetitions);
      check(cudaGetLastError(), "staged transform launch");
    }
    sync(streams);
  }
  double result = elapsed(start) / iterations;
  destroy(streams);
  return result;
}

static double pipeline(float *host, float *device, size_t chunk_bytes,
                       int elements, int chunks, int stream_count,
                       int repetitions, int iterations, int blocks, int threads,
                       bool staged) {
  std::vector<cudaStream_t> streams = makeStreams(staged ? stream_count : 1);
  auto start = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < chunks; ++c) {
      int stream_id = staged ? c % stream_count : 0;
      cudaStream_t stream = streams[stream_id];
      char *h = reinterpret_cast<char *>(host) + c * chunk_bytes;
      float *d = device + c * elements;

      // These three operations stay ordered in one stream. Other chunks can
      // progress concurrently in their own streams.
      check(cudaMemcpyAsync(d, h, chunk_bytes, cudaMemcpyHostToDevice, stream),
            "pipeline host-to-device copy");
      transform<<<blocks, threads, 0, stream>>>(d, elements, repetitions);
      check(cudaGetLastError(), "pipeline transform launch");
      check(cudaMemcpyAsync(h, d, chunk_bytes, cudaMemcpyDeviceToHost, stream),
            "pipeline device-to-host copy");
    }
    sync(streams);
  }
  double result = elapsed(start) / iterations;
  destroy(streams);
  return result;
}

int main(int argc, char **argv) {
  int chunks = argc > 1 ? std::atoi(argv[1]) : 8;
  int elements = argc > 2 ? std::atoi(argv[2]) : 1 << 20;
  int stream_count = argc > 3 ? std::atoi(argv[3]) : 4;
  int repetitions = argc > 4 ? std::atoi(argv[4]) : 100;
  int iterations = argc > 5 ? std::atoi(argv[5]) : 5;
  if (chunks <= 0 || elements <= 0 || stream_count <= 0 ||
      repetitions <= 0 || iterations <= 0) {
    std::fprintf(stderr, "all arguments must be positive\n");
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  check(cudaSetDevice(0), "cudaSetDevice");
  check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  size_t total_elements = static_cast<size_t>(chunks) * elements;
  size_t chunk_bytes = static_cast<size_t>(elements) * sizeof(float);
  float *host = nullptr;
  float *device = nullptr;

  // Pinned host memory is required for genuinely asynchronous host transfers.
  check(cudaHostAlloc(&host, total_elements * sizeof(float),
                      cudaHostAllocDefault),
        "cudaHostAlloc");
  check(cudaMalloc(&device, total_elements * sizeof(float)), "cudaMalloc");
  for (size_t i = 0; i < total_elements; ++i) {
    host[i] = 1.0f;
  }
  check(cudaMemcpy(device, host, total_elements * sizeof(float),
                   cudaMemcpyHostToDevice),
        "initial copy");
  check(cudaDeviceSynchronize(), "initial synchronization");

  int threads = 256;
  int blocks = (elements + threads - 1) / threads;
  // Warm up the kernel and CUDA runtime before timing. Without this launch,
  // lazy module loading and context setup could contaminate the first case.
  transform<<<blocks, threads>>>(device, elements, repetitions);
  check(cudaGetLastError(), "warm-up transform launch");
  check(cudaDeviceSynchronize(), "warm-up transform synchronization");
  check(cudaMemcpy(device, host, total_elements * sizeof(float),
                   cudaMemcpyHostToDevice),
        "post-warm-up initialization");
  double copy_one = copyOneStream(host, device, chunk_bytes, chunks, iterations);
  double copy_many = copyStaged(host, device, chunk_bytes, chunks, stream_count,
                                iterations);
  check(cudaMemcpy(device, host, total_elements * sizeof(float),
                   cudaMemcpyHostToDevice),
        "kernel initialization");
  double kernel_one = kernelOneStream(device, elements, chunks, repetitions,
                                      iterations, blocks, threads);
  double kernel_many = kernelStaged(device, elements, chunks, stream_count,
                                    repetitions, iterations, blocks, threads);
  double full_one = pipeline(host, device, chunk_bytes, elements, chunks, 1,
                             repetitions, iterations, blocks, threads, false);
  double full_many = pipeline(host, device, chunk_bytes, elements, chunks,
                              stream_count, repetitions, iterations, blocks,
                              threads, true);

  std::printf("GPU: %s\n", prop.name);
  std::printf("Chunks: %d, elements/chunk: %d, streams: %d\n", chunks, elements,
              stream_count);
  std::printf("Kernel repetitions: %d, measured iterations: %d\n", repetitions,
              iterations);
  std::printf("\nAverage end-to-end time per iteration:\n");
  std::printf("  Copy only, one stream:     %.3f ms\n", copy_one);
  std::printf("  Copy only, staged:         %.3f ms (%.2fx)\n", copy_many,
              copy_one / copy_many);
  std::printf("  Kernel only, one stream:   %.3f ms\n", kernel_one);
  std::printf("  Kernel only, staged:       %.3f ms (%.2fx)\n", kernel_many,
              kernel_one / kernel_many);
  std::printf("  Full pipeline, one stream: %.3f ms\n", full_one);
  std::printf("  Full pipeline, staged:     %.3f ms (%.2fx)\n", full_many,
              full_one / full_many);
  std::printf("\nAsynchronous copy engines: %d\n", prop.asyncEngineCount);
  std::printf("The staged result is meaningful only when the profiler shows "
              "overlap.\n");

  check(cudaFree(device), "cudaFree");
  check(cudaFreeHost(host), "cudaFreeHost");
  return EXIT_SUCCESS;
}
