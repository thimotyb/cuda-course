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

static int parseInt(const char *value, const char *name) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > 2147483647L) {
    std::fprintf(stderr, "Invalid %s: %s\n", name, value);
    std::exit(1);
  }
  return static_cast<int>(parsed);
}

static bool parseBoolFlag(const char *value, const char *name) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || (parsed != 0 && parsed != 1)) {
    std::fprintf(stderr, "Invalid %s: %s. Use 0 or 1.\n", name, value);
    std::exit(1);
  }
  return parsed != 0;
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

static bool isSupportedBlockSize(int block_size) {
  return block_size == 64 || block_size == 128 || block_size == 256 ||
         block_size == 512 || block_size == 1024;
}

static bool isSupportedRegisterLevel(int register_level) {
  return register_level == 8 || register_level == 16 || register_level == 32 ||
         register_level == 64 || register_level == 96 ||
         register_level == 128;
}

template <int RegisterLevel>
__global__ void registerPressureKernel(const float *input, float *output,
                                       unsigned long long n,
                                       int arithmetic_rounds) {
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float regs[RegisterLevel];
  float seed = input[i] + static_cast<float>(threadIdx.x & 31) * 0.001f;

#pragma unroll
  for (int r = 0; r < RegisterLevel; ++r) {
    regs[r] = seed + static_cast<float>(r) * 0.01f;
  }

  // The array has compile-time size and fixed-index unrolled access, so the
  // compiler usually scalarizes it into registers until register pressure is too
  // high. The actual register count is reported with cudaFuncGetAttributes().
  for (int round = 0; round < arithmetic_rounds; ++round) {
#pragma unroll
    for (int r = 0; r < RegisterLevel; ++r) {
      regs[r] = fmaf(regs[r], 1.000001f,
                     static_cast<float>((r % 7) + 1) * 0.000001f);
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int r = 0; r < RegisterLevel; ++r) {
    sum += regs[r];
  }

  output[i] = sum;
}

template <int RegisterLevel>
static void *kernelPtr() {
  return reinterpret_cast<void *>(&registerPressureKernel<RegisterLevel>);
}

template <int RegisterLevel>
static cudaFuncAttributes kernelAttributes() {
  cudaFuncAttributes attrs{};
  checkCuda(cudaFuncGetAttributes(&attrs, registerPressureKernel<RegisterLevel>),
            "cudaFuncGetAttributes failed");
  return attrs;
}

template <int RegisterLevel>
static void launchKernel(int blocks, int block_size, const float *device_input,
                         float *device_output, unsigned long long n,
                         int arithmetic_rounds) {
  registerPressureKernel<RegisterLevel>
      <<<blocks, block_size>>>(device_input, device_output, n,
                               arithmetic_rounds);
}

struct RunConfig {
  unsigned long long n;
  int block_size;
  int register_level;
  int arithmetic_rounds;
  int iterations;
};

struct RunResult {
  int requested_register_level;
  int actual_registers_per_thread;
  int block_size;
  int active_blocks_per_sm;
  int active_warps_per_sm;
  int max_warps_per_sm;
  float theoretical_occupancy_percent;
  float average_ms;
  double elements_per_second;
};

static long long registersPerBlock(int block_size, int registers_per_thread) {
  return static_cast<long long>(block_size) *
         static_cast<long long>(registers_per_thread);
}

static cudaFuncAttributes attributesForLevel(int register_level) {
  switch (register_level) {
  case 8:
    return kernelAttributes<8>();
  case 16:
    return kernelAttributes<16>();
  case 32:
    return kernelAttributes<32>();
  case 64:
    return kernelAttributes<64>();
  case 96:
    return kernelAttributes<96>();
  case 128:
    return kernelAttributes<128>();
  default:
    std::fprintf(stderr, "Unsupported register level: %d\n", register_level);
    std::exit(1);
  }
}

static void *kernelPtrForLevel(int register_level) {
  switch (register_level) {
  case 8:
    return kernelPtr<8>();
  case 16:
    return kernelPtr<16>();
  case 32:
    return kernelPtr<32>();
  case 64:
    return kernelPtr<64>();
  case 96:
    return kernelPtr<96>();
  case 128:
    return kernelPtr<128>();
  default:
    std::fprintf(stderr, "Unsupported register level: %d\n", register_level);
    std::exit(1);
  }
}

static void launchForLevel(int register_level, int blocks, int block_size,
                           const float *device_input, float *device_output,
                           unsigned long long n, int arithmetic_rounds) {
  switch (register_level) {
  case 8:
    launchKernel<8>(blocks, block_size, device_input, device_output, n,
                    arithmetic_rounds);
    break;
  case 16:
    launchKernel<16>(blocks, block_size, device_input, device_output, n,
                     arithmetic_rounds);
    break;
  case 32:
    launchKernel<32>(blocks, block_size, device_input, device_output, n,
                     arithmetic_rounds);
    break;
  case 64:
    launchKernel<64>(blocks, block_size, device_input, device_output, n,
                     arithmetic_rounds);
    break;
  case 96:
    launchKernel<96>(blocks, block_size, device_input, device_output, n,
                     arithmetic_rounds);
    break;
  case 128:
    launchKernel<128>(blocks, block_size, device_input, device_output, n,
                      arithmetic_rounds);
    break;
  default:
    std::fprintf(stderr, "Unsupported register level: %d\n", register_level);
    std::exit(1);
  }
}

static RunResult runOnce(const RunConfig &config, const cudaDeviceProp &prop,
                         const float *device_input, float *device_output) {
  if (!isSupportedBlockSize(config.block_size)) {
    std::fprintf(stderr,
                 "Unsupported block size: %d. Use 64, 128, 256, 512, or 1024.\n",
                 config.block_size);
    std::exit(1);
  }
  if (!isSupportedRegisterLevel(config.register_level)) {
    std::fprintf(stderr,
                 "Unsupported register level: %d. Use 8, 16, 32, 64, 96, or "
                 "128.\n",
                 config.register_level);
    std::exit(1);
  }
  if (config.block_size > prop.maxThreadsPerBlock) {
    std::fprintf(stderr,
                 "Block size %d exceeds device maxThreadsPerBlock %d.\n",
                 config.block_size, prop.maxThreadsPerBlock);
    std::exit(1);
  }

  unsigned long long blocks_ull =
      (config.n + static_cast<unsigned long long>(config.block_size) - 1ULL) /
      static_cast<unsigned long long>(config.block_size);
  if (blocks_ull > 2147483647ULL) {
    std::fprintf(stderr, "Grid is too large for this example: %llu blocks.\n",
                 blocks_ull);
    std::exit(1);
  }
  int blocks = static_cast<int>(blocks_ull);

  cudaFuncAttributes attrs = attributesForLevel(config.register_level);
  long long requested_registers_per_block =
      registersPerBlock(config.block_size, attrs.numRegs);
  if (requested_registers_per_block > prop.regsPerBlock) {
    std::fprintf(stderr,
                 "Configuration cannot launch: block_size=%d, actual "
                 "registers/thread=%d, registers/block=%lld, device "
                 "registers/block limit=%d.\n",
                 config.block_size, attrs.numRegs,
                 requested_registers_per_block, prop.regsPerBlock);
    std::exit(1);
  }

  int active_blocks_per_sm = 0;
  checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_sm, kernelPtrForLevel(config.register_level),
                config.block_size, 0),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed");
  if (active_blocks_per_sm == 0) {
    std::fprintf(stderr,
                 "Configuration has zero active blocks per SM: block_size=%d, "
                 "actual registers/thread=%d.\n",
                 config.block_size, attrs.numRegs);
    std::exit(1);
  }

  int active_warps_per_sm =
      active_blocks_per_sm * ((config.block_size + prop.warpSize - 1) /
                              prop.warpSize);
  int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  float theoretical_occupancy_percent =
      100.0f * static_cast<float>(active_warps_per_sm) /
      static_cast<float>(max_warps_per_sm);

  launchForLevel(config.register_level, blocks, config.block_size, device_input,
                 device_output, config.n, config.arithmetic_rounds);
  checkCuda(cudaGetLastError(), "warmup kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "warmup kernel failed");

  cudaEvent_t start;
  cudaEvent_t stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  checkCuda(cudaEventRecord(start), "cudaEventRecord start failed");
  for (int i = 0; i < config.iterations; ++i) {
    launchForLevel(config.register_level, blocks, config.block_size,
                   device_input, device_output, config.n,
                   config.arithmetic_rounds);
  }
  checkCuda(cudaGetLastError(), "timed kernel launch failed");
  checkCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

  float elapsed_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime failed");
  checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");

  RunResult result{};
  result.requested_register_level = config.register_level;
  result.actual_registers_per_thread = attrs.numRegs;
  result.block_size = config.block_size;
  result.active_blocks_per_sm = active_blocks_per_sm;
  result.active_warps_per_sm = active_warps_per_sm;
  result.max_warps_per_sm = max_warps_per_sm;
  result.theoretical_occupancy_percent = theoretical_occupancy_percent;
  result.average_ms = elapsed_ms / static_cast<float>(config.iterations);
  result.elements_per_second =
      static_cast<double>(config.n) / (static_cast<double>(result.average_ms) *
                                      1.0e-3);
  return result;
}

static void printHeader(const cudaDeviceProp &prop, const RunConfig &config) {
  std::printf("device: %s\n", prop.name);
  std::printf("SM count: %d\n", prop.multiProcessorCount);
  std::printf("warp size: %d\n", prop.warpSize);
  std::printf("max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
  std::printf("registers per block limit: %d\n", prop.regsPerBlock);
  std::printf("registers per SM: %d\n", prop.regsPerMultiprocessor);
  std::printf("N: %llu\n", config.n);
  std::printf("arithmetic rounds: %d\n", config.arithmetic_rounds);
  std::printf("iterations: %d\n", config.iterations);
}

static void printResult(const RunResult &result) {
  std::printf(
      "%10d %10d %12d %14d %12d %10d %10.1f %12.4f %12.2f\n",
      result.block_size, result.requested_register_level,
      result.actual_registers_per_thread, result.active_blocks_per_sm,
      result.active_warps_per_sm, result.max_warps_per_sm,
      result.theoretical_occupancy_percent, result.average_ms,
      result.elements_per_second / 1.0e9);
}

static void printTableTitle() {
  std::printf("\n%10s %10s %12s %14s %12s %10s %10s %12s %12s\n",
              "block", "level", "actual regs", "blocks/SM", "warps/SM",
              "max warps", "occ %", "avg ms", "Gelem/s");
}

int main(int argc, char **argv) {
  RunConfig config{};
  config.n = 1ULL << 24;
  config.block_size = 256;
  config.register_level = 32;
  config.arithmetic_rounds = 128;
  config.iterations = 20;
  bool sweep = false;

  if (argc > 1) {
    config.n = parseSize(argv[1], "N");
  }
  if (argc > 2) {
    config.block_size = parseInt(argv[2], "block size");
  }
  if (argc > 3) {
    config.register_level = parseInt(argv[3], "register level");
  }
  if (argc > 4) {
    config.arithmetic_rounds = parseInt(argv[4], "arithmetic rounds");
  }
  if (argc > 5) {
    config.iterations = parseInt(argv[5], "iterations");
  }
  if (argc > 6) {
    sweep = parseBoolFlag(argv[6], "sweep");
  }
  if (argc > 7) {
    std::fprintf(stderr,
                 "Usage: %s [N] [block_size] [register_level] "
                 "[arithmetic_rounds] [iterations] [sweep]\n",
                 argv[0]);
    return 1;
  }

  int device = 0;
  cudaDeviceProp prop{};
  checkCuda(cudaSetDevice(device), "cudaSetDevice failed");
  checkCuda(cudaGetDeviceProperties(&prop, device),
            "cudaGetDeviceProperties failed");

  size_t bytes = static_cast<size_t>(config.n) * sizeof(float);
  std::vector<float> host_input(static_cast<size_t>(config.n), 1.0f);
  std::vector<float> host_output(static_cast<size_t>(config.n), 0.0f);

  float *device_input = nullptr;
  float *device_output = nullptr;
  checkCuda(cudaMalloc(&device_input, bytes), "cudaMalloc input failed");
  checkCuda(cudaMalloc(&device_output, bytes), "cudaMalloc output failed");
  checkCuda(cudaMemcpy(device_input, host_input.data(), bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy input failed");

  printHeader(prop, config);
  printTableTitle();

  if (sweep) {
    const int block_sizes[] = {64, 128, 256, 512, 1024};
    const int register_levels[] = {8, 16, 32, 64, 96, 128};
    for (int block_size : block_sizes) {
      if (block_size > prop.maxThreadsPerBlock) {
        continue;
      }
      for (int register_level : register_levels) {
        cudaFuncAttributes attrs = attributesForLevel(register_level);
        long long regs_per_block = registersPerBlock(block_size, attrs.numRegs);
        if (regs_per_block > prop.regsPerBlock) {
          std::printf("skipping block=%d level=%d: %lld registers/block exceeds "
                      "device limit %d\n",
                      block_size, register_level, regs_per_block,
                      prop.regsPerBlock);
          continue;
        }
        RunConfig run_config = config;
        run_config.block_size = block_size;
        run_config.register_level = register_level;
        RunResult result =
            runOnce(run_config, prop, device_input, device_output);
        printResult(result);
      }
    }
  } else {
    RunResult result = runOnce(config, prop, device_input, device_output);
    printResult(result);
  }

  checkCuda(cudaMemcpy(host_output.data(), device_output, bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy output failed");
  if (!std::isfinite(host_output[0])) {
    std::fprintf(stderr, "Invalid output value.\n");
    return 1;
  }

  checkCuda(cudaFree(device_input), "cudaFree input failed");
  checkCuda(cudaFree(device_output), "cudaFree output failed");

  return 0;
}
