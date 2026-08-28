# PyTorch CUDA Tensors

This example introduces PyTorch tensors as the practical data structure used to run work on a CUDA GPU.

It follows the same ideas used in the PyTorch tensor tutorial, but makes CUDA placement explicit and compares a small tensor workload on CPU and on `cuda:0`.

## Requirements

- Python 3
- PyTorch with CUDA support
- NumPy, used by the PyTorch tensor tutorial and commonly installed with PyTorch exercises
- An NVIDIA GPU visible to PyTorch

This repository uses a local virtual environment for the Python exercises:

```bash
cd /home/thimoty/git/cuda-course
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy --index-url https://download.pytorch.org/whl/cu130
```

The `cu130` wheel targets CUDA 13.0 runtime libraries and is appropriate for systems with a sufficiently recent NVIDIA driver.

If `.venv` already exists, activate it before running the exercise:

```bash
cd /home/thimoty/git/cuda-course
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

After activation, `python` and `pip` refer to the virtual environment. If you do not want to activate it, call `.venv/bin/python` explicitly.

Check the environment:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Run

```bash
cd examples/ch08/pytorch_cuda_tensors
../../../.venv/bin/python pytorch_cuda_tensors.py
```

Optional matrix size:

```bash
../../../.venv/bin/python pytorch_cuda_tensors.py --size 4096
```

Run the benchmark at three sizes:

```bash
../../../.venv/bin/python pytorch_cuda_tensors.py --size 128
../../../.venv/bin/python pytorch_cuda_tensors.py --size 2048
../../../.venv/bin/python pytorch_cuda_tensors.py --size 4096
```

Record `CPU workload time`, `CUDA workload time`, and `CUDA speedup over CPU`.

The `128 x 128` case may be faster on CPU because CUDA work has fixed costs: GPU tensor allocation, kernel launch, scheduling through PyTorch and backend libraries, and synchronization before reading accurate timing. Larger matrices expose more parallel work, so those fixed costs are spread over more useful computation.

## Expected behavior

The program prints:

- the selected CUDA device;
- tensor shape, dtype, and device;
- examples of tensor initialization on CUDA;
- indexing, concatenation, element-wise operations, and matrix multiplication on CUDA;
- CUDA event timing for a matrix multiplication;
- CPU timing and CUDA timing for the same tensor workload;
- the observed CUDA speedup over CPU for that workload;
- current PyTorch CUDA memory usage.

If CUDA is not available, the program stops with a clear message instead of silently falling back to CPU execution.
