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

// Classic tree-based parallel reduction: each block reduces its own slice of
// `input` (one block_size-sized chunk) down to a single partial sum. main()
// then finishes the job by summing the (much smaller) per-block partials on
// the CPU -- a second, cheap reduction stage.
__global__ void reduceBlocksKernel(const float *input, float *partials,
                                   unsigned long long n) {
  // extern __shared__: the array's size is not known at compile time -- it
  // is set per launch via the third <<<...>>> argument (shared_bytes in
  // main(), sized to hold exactly block_size floats). Shared memory is
  // on-chip and shared by every thread in the block, unlike global memory.
  extern __shared__ float shared[];

  unsigned int tid = threadIdx.x;
  // Step 1: compute this thread's position in the whole input array. Each
  // block covers blockDim.x contiguous elements, so global_index is simply
  // the block's offset plus the thread's local index.
  unsigned long long global_index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + tid;

  // Step 2: load one element per thread from global memory into shared
  // memory. The last block may run past the end of `input` (n need not be a
  // multiple of block_size), so out-of-range threads contribute 0.0f instead
  // -- the neutral element for addition -- rather than reading garbage.
  shared[tid] = (global_index < n) ? input[global_index] : 0.0f;
  // Every thread must finish its write to `shared` before any thread starts
  // reading neighboring entries in the loop below.
  __syncthreads();

  // Step 3: tree reduction within the block. On each pass, the number of
  // active threads (and the stride between the pair each one adds) is
  // halved: pass 1 has half the threads add element (tid + blockDim.x/2)
  // into element tid, pass 2 halves that again, and so on down to stride 1.
  // After the loop shared[0] holds the sum of all block_size elements.
  // log2(blockDim.x) passes total, so doubling block_size adds only one
  // more pass -- reduction cost grows logarithmically, not linearly.
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    // Required before the next pass: threads that already finished adding
    // must not race ahead and read a shared[] slot another thread in this
    // pass hasn't written yet.
    __syncthreads();
  }

  // Step 4: exactly one thread per block (tid 0) writes this block's final
  // partial sum out to global memory, one float per block.
  if (tid == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

int main(int argc, char **argv) {
  // Step 1: defaults, then override from argv. N is the array size to sum
  // (default 2^26 ~= 67M floats), block_size is threads per block (which
  // this example restricts to the sizes the kernel/shared-memory layout
  // supports), iterations is how many timed kernel launches to average.
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
  // Step 2: block_size must be a power of two so the stride-halving loop in
  // the kernel divides evenly all the way down to 1 with no leftover
  // element. This is also where the "occupancy" angle of this example comes
  // in: block_size determines how many resident warps/threads each
  // streaming multiprocessor can host at once (together with the shared
  // memory the block requests, sized below), so trying different supported
  // values here is how you would explore the occupancy/performance tradeoff.
  if (!isSupportedBlockSize(block_size)) {
    std::fprintf(stderr,
                 "Unsupported block size: %d. Use 64, 128, 256, 512, or 1024.\n",
                 block_size);
    return 1;
  }

  // Step 3: query the device and make sure the requested block_size is
  // actually launchable on it (maxThreadsPerBlock is a hardware limit).
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

  // Step 4: size the grid so that blocks * block_size covers all N elements
  // (ceiling division -- the last block may be only partially full, which is
  // exactly why the kernel guards global_index against n). Also guard
  // against overflowing a 32-bit block count, which is what <<<blocks, ...>>>
  // actually takes.
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
  // One shared-memory float per thread in the block -- this is the dynamic
  // shared memory size the kernel's `extern __shared__ float shared[]`
  // needs, and it grows with block_size (more threads per block also means
  // more shared memory reserved per block, which is the other side of the
  // occupancy tradeoff besides thread count).
  size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

  // Step 5: build the input on the host (N floats, all 1.0f so the correct
  // sum is simply N) and compute a reference sum on the CPU in double
  // precision, to check the GPU result against later.
  std::vector<float> host_input(static_cast<size_t>(n), 1.0f);
  std::vector<float> host_partials(static_cast<size_t>(blocks), 0.0f);

  double cpu_sum = 0.0;
  for (float value : host_input) {
    cpu_sum += static_cast<double>(value);
  }

  // Step 6: allocate the device buffers and copy the input over. Only one
  // float per block comes back (device_partials), since each block already
  // reduces its whole chunk down to a single value.
  float *device_input = nullptr;
  float *device_partials = nullptr;
  checkCuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc input failed");
  checkCuda(cudaMalloc(&device_partials, partial_bytes),
            "cudaMalloc partials failed");
  checkCuda(cudaMemcpy(device_input, host_input.data(), input_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy input failed");

  // Step 7: warm-up launch, synchronized, so first-use CUDA context/driver
  // setup does not pollute the timed measurement below.
  reduceBlocksKernel<<<blocks, block_size, shared_bytes>>>(device_input,
                                                           device_partials, n);
  checkCuda(cudaGetLastError(), "warmup kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "warmup kernel failed");

  // Step 8: create CUDA events to time the kernel on the GPU's own clock.
  cudaEvent_t start;
  cudaEvent_t stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  // Step 9: launch the kernel `iterations` times back to back on the
  // (implicitly synchronizing) default stream, bracketed by start/stop
  // events, then average the total elapsed time per launch below.
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

  // Step 10: copy the per-block partial sums back and finish the reduction
  // on the CPU -- with `blocks` typically in the thousands rather than
  // millions, summing them serially here is cheap compared to reducing the
  // full input, so a second GPU pass is not needed for this example.
  checkCuda(cudaMemcpy(host_partials.data(), device_partials, partial_bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy partials failed");

  double gpu_sum = 0.0;
  for (float partial : host_partials) {
    gpu_sum += static_cast<double>(partial);
  }

  // Step 11: compare the GPU result against the CPU reference within a
  // small relative tolerance (floating-point addition is not perfectly
  // associative, so summing in a different order can differ slightly).
  double absolute_error = std::fabs(cpu_sum - gpu_sum);
  double tolerance = std::max(1.0, std::fabs(cpu_sum)) * 1.0e-5;
  bool pass = absolute_error <= tolerance;

  // Step 12: report configuration, correctness, and timing.
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

  // Step 13: release CUDA events and device buffers before exiting.
  checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
  checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");
  checkCuda(cudaFree(device_input), "cudaFree input failed");
  checkCuda(cudaFree(device_partials), "cudaFree partials failed");

  return pass ? 0 : 1;
}
