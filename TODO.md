# TODO

- Add a small Thrust demo to the CUDA examples. The demo should introduce Thrust as a C++ parallel algorithms library included with the NVIDIA CUDA Toolkit, similar in spirit to the C++ STL, and show how it can express common GPU operations without writing an explicit CUDA kernel launch.
  Suggested scope:
  - explain `thrust::host_vector` and `thrust::device_vector`;
  - show a vector addition example with `thrust::transform` and `thrust::plus`;
  - mention common algorithms such as `thrust::sort`, `thrust::stable_sort`, `thrust::find`, `thrust::binary_search`, `thrust::reduce`, `thrust::count`, and `thrust::transform`;
  - mention interoperability with CUDA C/C++ and libraries such as cuBLAS and cuFFT;
  - note that backend selection can target CUDA GPU execution or CPU backends such as OpenMP/TBB depending on configuration.

- Add a small CUDA-X deep learning demo. The demo should introduce cuDNN as the CUDA-X library for GPU-accelerated deep neural network primitives, and should avoid becoming a full training framework.
  Suggested scope:
  - explain that CUDA-X is a family of NVIDIA libraries built on CUDA, not one single library;
  - position cuDNN as the right CUDA-X library for DNN building blocks such as convolution, matmul, activation, pooling, normalization, and backward operations;
  - start with a minimal forward-pass demo, such as one convolution/ReLU step or one tiny MLP layer;
  - optionally add one training iteration only if the code stays readable;
  - consider a cuBLAS-first MLP example for `Y = XW + b`, then a cuDNN example for convolution/ReLU;
  - emphasize that cuDNN provides optimized primitives, while the program still owns tensor setup, memory, workspaces, weights, gradients, the optimizer, and the training loop.

- Add a pinned-memory transfer demo. The demo should compare ordinary pageable host memory against pinned/page-locked host memory and show that pinned memory can make large host-device transfers faster.
  Suggested scope:
  - add a short conceptual explanation of what pinned/page-locked host memory means and why the operating system cannot freely page or move it;
  - explain when pinned memory is useful: large host-device transfers, `cudaMemcpyAsync`, streams, and overlap between copies and kernel execution;
  - include a compact code example that shows the allocation, copy, timing, and cleanup flow end to end;
  - allocate the pageable version with standard host allocation, such as `std::vector<float>` or `malloc`;
  - allocate the pinned version with `cudaMallocHost` or `cudaHostAlloc`;
  - copy the same large buffer from host to device and device to host in both versions;
  - measure transfer time with CUDA events, averaging over multiple iterations;
  - print bandwidth in GB/s for pageable versus pinned memory;
  - optionally add a `cudaMemcpyAsync` plus stream version to show why pinned memory matters for asynchronous transfers and overlap;
  - include a warning that pinned memory should be used for transfer buffers, not as a default replacement for all host allocations.

- Add a topic on using GPUs for local inference and estimating inference cost per million tokens.
  Suggested scope:
  - explain the difference between local inference on owned hardware, rented GPU inference, and managed API pricing;
  - introduce tokens/second per user, batch size, utilization, model size, quantization, memory capacity, and power cost as the main variables;
  - include a worked example for H100-class inference cost per million tokens;
  - add a practical vLLM section showing how to host an open LLM locally or on a rented GPU, expose an OpenAI-compatible endpoint, and observe GPU memory use, throughput, batching, and latency;
  - candidate reference point to verify before publishing: NVIDIA H100 GPU FAQs report inference at approximately `$0.09` per million tokens at `66 TPS/user` for GPT-OSS-120B using vLLM, based on SemiAnalysis InferenceX benchmarks as of April 2026;
  - compare this with a local Blackwell consumer GPU example, clearly separating hardware purchase cost, electricity, amortization, and achievable throughput.
