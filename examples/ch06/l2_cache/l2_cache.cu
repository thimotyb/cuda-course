#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// Each thread reads one element. Repeating the same working set creates
// temporal reuse, which is the situation an L2 persisting hint can target.
__global__ void readWorkingSet(const float *data, float *result, int elements,
                               int repetitions) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    float value = 0.0f;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      value += data[index];
    }
    result[index] = value;
  }
}

// >>> This is where the L2 access-policy hint is actually built and attached.
// The accessPolicyWindow tells the driver: for accesses to [base_ptr,
// base_ptr + num_bytes) on this stream, treat hitRatio of them as
// hit_property (e.g. cudaAccessPropertyPersisting, to favor keeping this
// range resident in the set-aside L2 partition) and the rest as
// miss_property (e.g. cudaAccessPropertyStreaming, i.e. don't bother
// persisting them). cudaStreamSetAttribute() with
// cudaStreamAttributeAccessPolicyWindow is the call that installs the hint
// on the stream; every kernel launched on `stream` afterward is subject to it.
static void setPolicy(cudaStream_t stream, const float *data, size_t bytes,
                      cudaAccessProperty hit_property,
                      cudaAccessProperty miss_property, float hit_ratio) {
  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.base_ptr =
      const_cast<float *>(data);              // hint window start
  attribute.accessPolicyWindow.num_bytes = bytes;   // hint window size
  attribute.accessPolicyWindow.hitRatio = hit_ratio; // fraction treated as "hit"
  attribute.accessPolicyWindow.hitProp = hit_property;   // e.g. Persisting
  attribute.accessPolicyWindow.missProp = miss_property; // e.g. Streaming
  check(cudaStreamSetAttribute(stream,
                              cudaStreamAttributeAccessPolicyWindow,  // <-- the L2 hint API
                              &attribute),
        "cudaStreamSetAttribute");
}

static float measure(const float *data, float *result, int elements,
                     int repetitions, int iterations,
                     cudaAccessProperty hit_property,
                     cudaAccessProperty miss_property, bool use_policy,
                     size_t policy_bytes) {
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
  check(cudaEventCreate(&start), "cudaEventCreate(start)");
  check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  if (use_policy) {
    // Installs the L2 access-policy hint on `stream` before any kernel runs.
    setPolicy(stream, data, policy_bytes, hit_property, miss_property, 1.0f);
  }

  // Warm up this exact policy before recording timings.
  readWorkingSet<<<(elements + 255) / 256, 256, 0, stream>>>(
      data, result, elements, repetitions);
  check(cudaGetLastError(), "warm-up launch");
  check(cudaStreamSynchronize(stream), "warm-up synchronization");

  check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    readWorkingSet<<<(elements + 255) / 256, 256, 0, stream>>>(
        data, result, elements, repetitions);
    check(cudaGetLastError(), "timed launch");
  }
  check(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
  check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, start, stop),
        "cudaEventElapsedTime");
  check(cudaEventDestroy(start), "cudaEventDestroy(start)");
  check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return milliseconds / iterations;
}

int main(int argc, char **argv) {
  int elements = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
  int repetitions = argc > 2 ? std::atoi(argv[2]) : 1000;
  int iterations = argc > 3 ? std::atoi(argv[3]) : 5;
  if (elements <= 0 || repetitions <= 0 || iterations <= 0) {
    std::fprintf(stderr, "all arguments must be positive\n");
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  check(cudaSetDevice(0), "cudaSetDevice");
  check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  size_t bytes = static_cast<size_t>(elements) * sizeof(float);
  float *data = nullptr;
  float *result = nullptr;
  check(cudaMalloc(&data, bytes), "cudaMalloc(data)");
  check(cudaMalloc(&result, bytes), "cudaMalloc(result)");
  std::vector<float> host_data(elements, 1.0f);
  check(cudaMemcpy(data, host_data.data(), bytes, cudaMemcpyHostToDevice),
        "initial data copy");
  check(cudaMemset(result, 0, bytes), "cudaMemset(result)");

  int blocks = (elements + 255) / 256;
  readWorkingSet<<<blocks, 256>>>(data, result, elements, 1);
  check(cudaGetLastError(), "initial warm-up launch");
  check(cudaDeviceSynchronize(), "initial warm-up synchronization");

  // Baseline: no access-policy hint at all (use_policy=false).
  float normal = measure(data, result, elements, repetitions, iterations,
                         cudaAccessPropertyNormal,
                         cudaAccessPropertyNormal, false, 0);
  // Hint present but explicitly telling L2 not to persist this range
  // (both hit and miss are Streaming) — isolates the hint mechanism's own
  // overhead from any benefit of actually persisting data.
  float streaming = measure(data, result, elements, repetitions, iterations,
                            cudaAccessPropertyStreaming,
                            cudaAccessPropertyStreaming, true, bytes);

  // The access-policy window cannot exceed the device's max window size, so
  // the hint's target range is clamped here before it is requested below.
  size_t policy_bytes = bytes;
  if (prop.accessPolicyMaxWindowSize > 0) {
    policy_bytes = policy_bytes < prop.accessPolicyMaxWindowSize
                       ? policy_bytes
                       : prop.accessPolicyMaxWindowSize;
  }
  // A device only honors the persisting hint if it both allows an
  // access-policy window and has L2 capacity set aside for persistence.
  bool supports_policy = prop.accessPolicyMaxWindowSize > 0 &&
                         prop.persistingL2CacheMaxSize > 0;
  float persisting = 0.0f;
  if (supports_policy) {
    // The actual "use L2 to persist this working set" measurement: hits are
    // Persisting (favor residency in the set-aside L2), misses fall back to
    // Streaming.
    persisting = measure(data, result, elements, repetitions, iterations,
                         cudaAccessPropertyPersisting,
                         cudaAccessPropertyStreaming, true, policy_bytes);
  }

  float checksum = 0.0f;
  check(cudaMemcpy(&checksum, result, sizeof(float), cudaMemcpyDeviceToHost),
        "checksum copy");
  std::printf("GPU: %s\n", prop.name);
  std::printf("Working set: %.2f MiB, repetitions: %d, iterations: %d\n",
              bytes / (1024.0 * 1024.0), repetitions, iterations);
  std::printf("L2 cache: %.2f MiB\n",
              prop.l2CacheSize / (1024.0 * 1024.0));
  std::printf("Maximum L2 set-aside: %.2f MiB\n",
              prop.persistingL2CacheMaxSize / (1024.0 * 1024.0));
  std::printf("Access-policy window: %s\n",
              supports_policy ? "supported" : "not supported");
  std::printf("\nAverage kernel time per iteration:\n");
  std::printf("  Normal:     %.3f ms\n", normal);
  std::printf("  Streaming:  %.3f ms\n", streaming);
  if (supports_policy) {
    std::printf("  Persisting: %.3f ms\n", persisting);
  } else {
    std::printf("  Persisting: skipped\n");
  }
  std::printf("Checksum sample: %.3f\n", checksum);

  check(cudaFree(result), "cudaFree(result)");
  check(cudaFree(data), "cudaFree(data)");
  return EXIT_SUCCESS;
}
