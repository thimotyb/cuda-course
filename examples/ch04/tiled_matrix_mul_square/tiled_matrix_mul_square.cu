#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

// This file is the short teaching version of tiled matrix multiplication.
// It deliberately supports only square matrices whose Width is a multiple of
// TILE_WIDTH, so the kernel can focus on tiling and shared memory without the
// extra boundary checks needed in production code.

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

// Square tiled matrix multiplication: P = M * N.
// This is the teaching version without boundary checks. Width must be a
// multiple of TILE_WIDTH, so every thread loads valid M and N elements.
__global__ void matrixMulKernel(const float *M, const float *N, float *P,
                                int Width) {
  // Shared memory is allocated once per block. All threads in the block
  // cooperate to fill these two tiles, then reuse them for TILE_WIDTH multiply
  // accumulate steps.
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // Short names match the notation used in the course material and make the
  // index formulas easier to compare with the diagrams.
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Each block computes one TILE_WIDTH x TILE_WIDTH output tile. Each thread
  // computes exactly one output element P[Row, Col].
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  // Pvalue is private to one thread. In a typical compilation it lives in a
  // register and accumulates the dot product for one output element.
  float Pvalue = 0.0f;

  // ph is the phase index. Each phase loads one tile from M and one tile from N.
  // Because Width is a multiple of TILE_WIDTH, Width / TILE_WIDTH is exact.
  for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
    // Cooperative tile loading:
    // - thread (ty, tx) loads one M element into Mds[ty][tx]
    // - thread (ty, tx) loads one N element into Nds[ty][tx]
    // No boundary checks are needed in this version because invalid edge tiles
    // are rejected by the host-side Width % TILE_WIDTH check.
    Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

    // Read-after-write synchronization: every thread must wait until the full
    // M and N tiles are present in shared memory before any thread consumes them.
    __syncthreads();

    // Reuse the two shared-memory tiles. Each thread walks across one row of
    // Mds and one column of Nds, accumulating TILE_WIDTH products.
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }

    // Write-after-read synchronization: no thread may overwrite Mds/Nds with
    // the next phase's tiles until all threads are done reading the current ones.
    __syncthreads();
  }

  // Since all output coordinates are valid in this square teaching version, the
  // final store also needs no boundary check.
  P[Row * Width + Col] = Pvalue;
}

// CPU reference implementation used only for correctness checking. It is not
// timed, and it intentionally uses the direct triple-loop form for clarity.
static void cpuMatrixMul(const std::vector<float> &M,
                         const std::vector<float> &N, std::vector<float> &P,
                         int Width) {
  for (int row = 0; row < Width; ++row) {
    for (int col = 0; col < Width; ++col) {
      float value = 0.0f;
      for (int k = 0; k < Width; ++k) {
        value += M[row * Width + k] * N[k * Width + col];
      }
      P[row * Width + col] = value;
    }
  }
}

static double maxAbsError(const std::vector<float> &expected,
                          const std::vector<float> &actual) {
  double max_error = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    double error = std::fabs(static_cast<double>(expected[i]) -
                             static_cast<double>(actual[i]));
    if (error > max_error) {
      max_error = error;
    }
  }
  return max_error;
}

int main(int argc, char **argv) {
  int Width = 512;
  int iterations = 20;

  if (argc > 1) {
    Width = parseInt(argv[1], "Width");
  }
  if (argc > 2) {
    iterations = parseInt(argv[2], "iterations");
  }
  if (argc > 3) {
    std::fprintf(stderr, "Usage: %s [Width] [iterations]\n", argv[0]);
    return 1;
  }

  if (Width % TILE_WIDTH != 0) {
    // This keeps the kernel simple enough for a first tiling explanation. A
    // production kernel would keep this case and add boundary checks instead.
    std::fprintf(stderr,
                 "Width must be a multiple of TILE_WIDTH. Width=%d, "
                 "TILE_WIDTH=%d.\n",
                 Width, TILE_WIDTH);
    return 1;
  }

  int device = 0;
  cudaDeviceProp prop{};
  checkCuda(cudaSetDevice(device), "cudaSetDevice failed");
  checkCuda(cudaGetDeviceProperties(&prop, device),
            "cudaGetDeviceProperties failed");

  size_t elements = static_cast<size_t>(Width) * static_cast<size_t>(Width);
  size_t bytes = elements * sizeof(float);

  std::vector<float> host_M(elements);
  std::vector<float> host_N(elements);
  std::vector<float> host_P(elements, 0.0f);
  std::vector<float> host_expected(elements, 0.0f);

  // Deterministic values make runs reproducible and avoid storing input files.
  for (int row = 0; row < Width; ++row) {
    for (int col = 0; col < Width; ++col) {
      host_M[row * Width + col] = static_cast<float>((row + col) % 13) * 0.25f;
      host_N[row * Width + col] = static_cast<float>((row - col + 17) % 11) * 0.5f;
    }
  }

  float *device_M = nullptr;
  float *device_N = nullptr;
  float *device_P = nullptr;
  checkCuda(cudaMalloc(&device_M, bytes), "cudaMalloc M failed");
  checkCuda(cudaMalloc(&device_N, bytes), "cudaMalloc N failed");
  checkCuda(cudaMalloc(&device_P, bytes), "cudaMalloc P failed");

  checkCuda(cudaMemcpy(device_M, host_M.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy M failed");
  checkCuda(cudaMemcpy(device_N, host_N.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy N failed");

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);

  // Warm up once before timing so one-time setup does not dominate the average.
  matrixMulKernel<<<dimGrid, dimBlock>>>(device_M, device_N, device_P, Width);
  checkCuda(cudaGetLastError(), "warmup kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "warmup kernel failed");

  cudaEvent_t start;
  cudaEvent_t stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  // CUDA events measure elapsed time on the GPU timeline. Host allocation,
  // host-device copies, and CPU verification are outside the timed region.
  checkCuda(cudaEventRecord(start), "cudaEventRecord start failed");
  for (int i = 0; i < iterations; ++i) {
    matrixMulKernel<<<dimGrid, dimBlock>>>(device_M, device_N, device_P, Width);
  }
  checkCuda(cudaGetLastError(), "timed kernel launch failed");
  checkCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

  float elapsed_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime failed");
  float average_ms = elapsed_ms / static_cast<float>(iterations);

  checkCuda(cudaMemcpy(host_P.data(), device_P, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy P failed");

  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpuMatrixMul(host_M, host_N, host_expected, Width);
  auto cpu_stop = std::chrono::high_resolution_clock::now();
  double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

  double error = maxAbsError(host_expected, host_P);
  bool pass = error < 1.0e-3 * static_cast<double>(Width);

  // A dense square matrix multiply performs Width^3 multiply-add pairs, counted
  // here as 2 * Width^3 floating-point operations.
  double flops = 2.0 * static_cast<double>(Width) *
                 static_cast<double>(Width) * static_cast<double>(Width);
  double gflops = flops / (static_cast<double>(average_ms) * 1.0e-3) / 1.0e9;
  double cpu_gflops = flops / (cpu_ms * 1.0e-3) / 1.0e9;
  double speedup = cpu_ms / static_cast<double>(average_ms);
  size_t shared_bytes_per_block =
      2ULL * TILE_WIDTH * TILE_WIDTH * sizeof(float);

  std::printf("device: %s\n", prop.name);
  std::printf("Width: %d\n", Width);
  std::printf("TILE_WIDTH: %d\n", TILE_WIDTH);
  std::printf("grid: (%u, %u, %u)\n", dimGrid.x, dimGrid.y, dimGrid.z);
  std::printf("block: (%u, %u, %u)\n", dimBlock.x, dimBlock.y, dimBlock.z);
  std::printf("shared memory per block: %zu bytes\n", shared_bytes_per_block);
  std::printf("iterations: %d\n", iterations);
  std::printf("average GPU kernel time: %.6f ms\n", average_ms);
  std::printf("GPU kernel throughput: %.2f GFLOP/s\n", gflops);
  std::printf("CPU triple-loop time: %.6f ms\n", cpu_ms);
  std::printf("CPU triple-loop throughput: %.2f GFLOP/s\n", cpu_gflops);
  std::printf("GPU kernel speedup vs CPU triple loop: %.2fx\n", speedup);
  std::printf("max absolute error: %.6e\n", error);
  std::printf("%s\n", pass ? "PASS" : "FAIL");

  checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
  checkCuda(cudaFree(device_M), "cudaFree M failed");
  checkCuda(cudaFree(device_N), "cudaFree N failed");
  checkCuda(cudaFree(device_P), "cudaFree P failed");

  return pass ? 0 : 1;
}
