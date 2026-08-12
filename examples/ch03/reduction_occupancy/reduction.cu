#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t result, const char *message) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(result));
    std::exit(1);
  }
}

static unsigned long long parseSize(const char *value, const char *name) {
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    std::fprintf(stderr, "Invalid %s: %s\n", name, value);
    std::exit(1);
  }
  return parsed;
}

static int parseInt(const char *value, const char *name) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > 2147483647L) {
    std::fprintf(stderr, "Invalid %s: %s\n", name, value);
    std::exit(1);
  }
  return static_cast<int>(parsed);
}

static bool isSupportedBlockSize(int block_size) {
  return block_size == 64 || block_size == 128 || block_size == 256 ||
         block_size == 512 || block_size == 1024;
}

__global__ void reduceBlocksKernel(const float *input, float *partials,
                                   unsigned long long n) {
  extern __shared__ float shared[];

  unsigned int tid = threadIdx.x;
  unsigned long long global_index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + tid;

  shared[tid] = (global_index < n) ? input[global_index] : 0.0f;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

int main(int argc, char **argv) {
  unsigned long long n = 1ULL << 26;
  int block_size = 256;
  int iterations = 20;

  if (argc > 1) {
    n = parseSize(argv[1], "N");
  }
  if (argc > 2) {
    block_size = parseInt(argv[2], "block size");
  }
  if (argc > 3) {
    iterations = parseInt(argv[3], "iterations");
  }
  if (argc > 4) {
    std::fprintf(stderr, "Usage: %s [N] [block_size] [iterations]\n", argv[0]);
    return 1;
  }
  if (!isSupportedBlockSize(block_size)) {
    std::fprintf(stderr,
                 "Unsupported block size: %d. Use 64, 128, 256, 512, or 1024.\n",
                 block_size);
    return 1;
  }

  int device = 0;
  cudaDeviceProp prop{};
  checkCuda(cudaGetDeviceProperties(&prop, device),
            "cudaGetDeviceProperties failed");
  if (block_size > prop.maxThreadsPerBlock) {
    std::fprintf(stderr,
                 "Block size %d exceeds device maxThreadsPerBlock %d.\n",
                 block_size, prop.maxThreadsPerBlock);
    return 1;
  }

  unsigned long long blocks_ull =
      (n + static_cast<unsigned long long>(block_size) - 1ULL) /
      static_cast<unsigned long long>(block_size);
  if (blocks_ull > 2147483647ULL) {
    std::fprintf(stderr, "Grid is too large for this example: %llu blocks.\n",
                 blocks_ull);
    return 1;
  }

  int blocks = static_cast<int>(blocks_ull);
  size_t input_bytes = static_cast<size_t>(n) * sizeof(float);
  size_t partial_bytes = static_cast<size_t>(blocks) * sizeof(float);
  size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

  std::vector<float> host_input(static_cast<size_t>(n), 1.0f);
  std::vector<float> host_partials(static_cast<size_t>(blocks), 0.0f);

  double cpu_sum = 0.0;
  for (float value : host_input) {
    cpu_sum += static_cast<double>(value);
  }

  float *device_input = nullptr;
  float *device_partials = nullptr;
  checkCuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc input failed");
  checkCuda(cudaMalloc(&device_partials, partial_bytes),
            "cudaMalloc partials failed");
  checkCuda(cudaMemcpy(device_input, host_input.data(), input_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy input failed");

  reduceBlocksKernel<<<blocks, block_size, shared_bytes>>>(device_input,
                                                           device_partials, n);
  checkCuda(cudaGetLastError(), "warmup kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "warmup kernel failed");

  cudaEvent_t start;
  cudaEvent_t stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  checkCuda(cudaEventRecord(start), "cudaEventRecord start failed");
  for (int i = 0; i < iterations; ++i) {
    reduceBlocksKernel<<<blocks, block_size, shared_bytes>>>(device_input,
                                                             device_partials, n);
  }
  checkCuda(cudaGetLastError(), "timed kernel launch failed");
  checkCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

  float elapsed_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime failed");

  checkCuda(cudaMemcpy(host_partials.data(), device_partials, partial_bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy partials failed");

  double gpu_sum = 0.0;
  for (float partial : host_partials) {
    gpu_sum += static_cast<double>(partial);
  }

  double absolute_error = std::fabs(cpu_sum - gpu_sum);
  double tolerance = std::max(1.0, std::fabs(cpu_sum)) * 1.0e-5;
  bool pass = absolute_error <= tolerance;

  std::printf("device: %s\n", prop.name);
  std::printf("N: %llu\n", n);
  std::printf("block size: %d\n", block_size);
  std::printf("blocks: %d\n", blocks);
  std::printf("shared memory per block: %zu bytes\n", shared_bytes);
  std::printf("iterations: %d\n", iterations);
  std::printf("CPU sum: %.6f\n", cpu_sum);
  std::printf("GPU sum: %.6f\n", gpu_sum);
  std::printf("absolute error: %.6f\n", absolute_error);
  std::printf("average kernel time: %.6f ms\n", elapsed_ms / iterations);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
  checkCuda(cudaFree(device_input), "cudaFree input failed");
  checkCuda(cudaFree(device_partials), "cudaFree partials failed");

  return pass ? 0 : 1;
}
