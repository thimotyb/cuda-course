#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

// Compare a teaching tiled matrix multiplication kernel with cuBLAS SGEMM.
//
// Both paths compute the same row-major square matrix product:
//
//   P = M * N
//
// The custom kernel is intentionally simple and mirrors the M4 tiling example.
// cuBLAS is a production library implementation and may use architecture-specific
// kernels, Tensor Core paths, scheduling strategies, and internal tiling that are
// far beyond this introductory kernel.

static void checkCuda(cudaError_t result, const char *message) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(result));
    std::exit(1);
  }
}

static void checkCublas(cublasStatus_t result, const char *message) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s: cuBLAS status %d\n", message,
                 static_cast<int>(result));
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

static bool usePedanticMath(int argc, char **argv) {
  if (argc <= 3) {
    return false;
  }
  if (std::strcmp(argv[3], "default") == 0) {
    return false;
  }
  if (std::strcmp(argv[3], "pedantic") == 0) {
    return true;
  }
  std::fprintf(stderr, "Invalid math mode: %s. Use default or pedantic.\n",
               argv[3]);
  std::exit(1);
}

__global__ void tiledMatrixMulKernel(const float *M, const float *N, float *P,
                                     int Width) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = blockIdx.y * TILE_WIDTH + ty;
  int Col = blockIdx.x * TILE_WIDTH + tx;

  float Pvalue = 0.0f;

  for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
    Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  P[Row * Width + Col] = Pvalue;
}

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

static float timeTiledKernel(const float *device_M, const float *device_N,
                             float *device_P, int Width, int iterations) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);

  tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(device_M, device_N, device_P,
                                              Width);
  checkCuda(cudaGetLastError(), "tiled warmup launch failed");
  checkCuda(cudaDeviceSynchronize(), "tiled warmup failed");

  cudaEvent_t start;
  cudaEvent_t stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate tiled start failed");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate tiled stop failed");

  checkCuda(cudaEventRecord(start), "cudaEventRecord tiled start failed");
  for (int i = 0; i < iterations; ++i) {
    tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(device_M, device_N, device_P,
                                                Width);
  }
  checkCuda(cudaGetLastError(), "tiled timed launch failed");
  checkCuda(cudaEventRecord(stop), "cudaEventRecord tiled stop failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize tiled failed");

  float elapsed_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime tiled failed");
  checkCuda(cudaEventDestroy(start), "cudaEventDestroy tiled start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy tiled stop failed");

  return elapsed_ms / static_cast<float>(iterations);
}

static float timeCublasSgemm(cublasHandle_t handle, const float *device_M,
                             const float *device_N, float *device_P, int Width,
                             int iterations) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // cuBLAS assumes column-major matrices. Our buffers are row-major.
  //
  // Row-major M is the same memory layout as column-major M^T. Therefore:
  //
  //   P_row = M_row * N_row
  //
  // can be computed by cuBLAS as:
  //
  //   P_col = N_col * M_col
  //
  // using the same memory buffers and swapping the M and N operands.
  checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Width, Width,
                          Width, &alpha, device_N, Width, device_M, Width,
                          &beta, device_P, Width),
              "cuBLAS warmup SGEMM failed");
  checkCuda(cudaDeviceSynchronize(), "cuBLAS warmup failed");

  cudaEvent_t start;
  cudaEvent_t stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate cuBLAS start failed");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate cuBLAS stop failed");

  checkCuda(cudaEventRecord(start), "cudaEventRecord cuBLAS start failed");
  for (int i = 0; i < iterations; ++i) {
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Width, Width,
                            Width, &alpha, device_N, Width, device_M, Width,
                            &beta, device_P, Width),
                "cuBLAS timed SGEMM failed");
  }
  checkCuda(cudaEventRecord(stop), "cudaEventRecord cuBLAS stop failed");
  checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize cuBLAS failed");

  float elapsed_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime cuBLAS failed");
  checkCuda(cudaEventDestroy(start), "cudaEventDestroy cuBLAS start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy cuBLAS stop failed");

  return elapsed_ms / static_cast<float>(iterations);
}

int main(int argc, char **argv) {
  int Width = 1024;
  int iterations = 20;

  if (argc > 1) {
    Width = parseInt(argv[1], "Width");
  }
  if (argc > 2) {
    iterations = parseInt(argv[2], "iterations");
  }
  if (argc > 4) {
    std::fprintf(stderr, "Usage: %s [Width] [iterations] [default|pedantic]\n",
                 argv[0]);
    return 1;
  }
  bool pedantic_math = usePedanticMath(argc, argv);

  if (Width % TILE_WIDTH != 0) {
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
  std::vector<float> host_tiled(elements, 0.0f);
  std::vector<float> host_cublas(elements, 0.0f);
  std::vector<float> host_expected(elements, 0.0f);

  for (int row = 0; row < Width; ++row) {
    for (int col = 0; col < Width; ++col) {
      host_M[row * Width + col] = static_cast<float>((row + col) % 13) * 0.25f;
      host_N[row * Width + col] =
          static_cast<float>((row - col + 17) % 11) * 0.5f;
    }
  }

  float *device_M = nullptr;
  float *device_N = nullptr;
  float *device_tiled = nullptr;
  float *device_cublas = nullptr;
  checkCuda(cudaMalloc(&device_M, bytes), "cudaMalloc M failed");
  checkCuda(cudaMalloc(&device_N, bytes), "cudaMalloc N failed");
  checkCuda(cudaMalloc(&device_tiled, bytes), "cudaMalloc tiled P failed");
  checkCuda(cudaMalloc(&device_cublas, bytes), "cudaMalloc cuBLAS P failed");

  checkCuda(cudaMemcpy(device_M, host_M.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy M failed");
  checkCuda(cudaMemcpy(device_N, host_N.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy N failed");

  cublasHandle_t handle = nullptr;
  checkCublas(cublasCreate(&handle), "cublasCreate failed");
  checkCublas(cublasSetMathMode(handle, pedantic_math ? CUBLAS_PEDANTIC_MATH
                                                      : CUBLAS_DEFAULT_MATH),
              "cublasSetMathMode failed");

  float tiled_ms =
      timeTiledKernel(device_M, device_N, device_tiled, Width, iterations);
  float cublas_ms =
      timeCublasSgemm(handle, device_M, device_N, device_cublas, Width,
                      iterations);

  checkCuda(cudaMemcpy(host_tiled.data(), device_tiled, bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy tiled P failed");
  checkCuda(cudaMemcpy(host_cublas.data(), device_cublas, bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy cuBLAS P failed");

  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpuMatrixMul(host_M, host_N, host_expected, Width);
  auto cpu_stop = std::chrono::high_resolution_clock::now();
  double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

  double tiled_error = maxAbsError(host_expected, host_tiled);
  double cublas_error = maxAbsError(host_expected, host_cublas);

  double flops = 2.0 * static_cast<double>(Width) *
                 static_cast<double>(Width) * static_cast<double>(Width);
  double tiled_gflops = flops / (static_cast<double>(tiled_ms) * 1.0e-3) / 1.0e9;
  double cublas_gflops =
      flops / (static_cast<double>(cublas_ms) * 1.0e-3) / 1.0e9;
  double cpu_gflops = flops / (cpu_ms * 1.0e-3) / 1.0e9;
  double cublas_vs_tiled = static_cast<double>(tiled_ms) /
                           static_cast<double>(cublas_ms);
  double cublas_vs_cpu = cpu_ms / static_cast<double>(cublas_ms);

  // Default cuBLAS math may use TF32 on supported NVIDIA GPUs. The tolerance is
  // therefore intentionally looser than the custom FP32 kernel tolerance.
  double cublas_tolerance =
      pedantic_math ? 1.0e-3 * static_cast<double>(Width)
                    : 5.0e-2 * static_cast<double>(Width);
  bool tiled_pass = tiled_error < 1.0e-3 * static_cast<double>(Width);
  bool cublas_pass = cublas_error < cublas_tolerance;

  std::printf("device: %s\n", prop.name);
  std::printf("Width: %d\n", Width);
  std::printf("TILE_WIDTH: %d\n", TILE_WIDTH);
  std::printf("iterations: %d\n", iterations);
  std::printf("cuBLAS math mode: %s\n",
              pedantic_math ? "pedantic FP32" : "default optimized");
  std::printf("CPU triple-loop time: %.6f ms\n", cpu_ms);
  std::printf("CPU triple-loop throughput: %.2f GFLOP/s\n", cpu_gflops);
  std::printf("tiled kernel time: %.6f ms\n", tiled_ms);
  std::printf("tiled kernel throughput: %.2f GFLOP/s\n", tiled_gflops);
  std::printf("cuBLAS SGEMM time: %.6f ms\n", cublas_ms);
  std::printf("cuBLAS SGEMM throughput: %.2f GFLOP/s\n", cublas_gflops);
  std::printf("cuBLAS speedup vs tiled kernel: %.2fx\n", cublas_vs_tiled);
  std::printf("cuBLAS speedup vs CPU triple loop: %.2fx\n", cublas_vs_cpu);
  std::printf("tiled max absolute error: %.6e\n", tiled_error);
  std::printf("cuBLAS max absolute error: %.6e\n", cublas_error);
  std::printf("%s\n", (tiled_pass && cublas_pass) ? "PASS" : "FAIL");

  checkCublas(cublasDestroy(handle), "cublasDestroy failed");
  checkCuda(cudaFree(device_M), "cudaFree M failed");
  checkCuda(cudaFree(device_N), "cudaFree N failed");
  checkCuda(cudaFree(device_tiled), "cudaFree tiled P failed");
  checkCuda(cudaFree(device_cublas), "cudaFree cuBLAS P failed");

  return (tiled_pass && cublas_pass) ? 0 : 1;
}
