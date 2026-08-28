# torch.cuda Overview

This example introduces the most useful `torch.cuda` commands for a first PyTorch GPU workflow.

It is not a deep-learning model. It is a small inspection and timing program that shows how PyTorch sees CUDA devices, how to check memory usage, and how to time GPU work correctly.

## Requirements

- Python 3
- PyTorch with CUDA support
- NumPy
- An NVIDIA GPU visible to PyTorch

From the repository root, create the shared virtual environment if it does not exist:

```bash
cd /home/thimoty/git/cuda-course
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy --index-url https://download.pytorch.org/whl/cu130
```

If `.venv` already exists, activate it:

```bash
cd /home/thimoty/git/cuda-course
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Run

From the repository root:

```bash
.venv/bin/python examples/ch08/torch_cuda_overview/torch_cuda_overview.py
```

Optional matrix size:

```bash
.venv/bin/python examples/ch08/torch_cuda_overview/torch_cuda_overview.py --size 4096
```

## Expected behavior

The program prints:

- PyTorch and CUDA runtime versions;
- whether CUDA is available;
- visible device count;
- current device index and name;
- selected device properties;
- current and default stream information;
- CUDA memory usage before and after allocation;
- CUDA event timing for matrix multiplication;
- a short example of synchronization before reading GPU results from Python.
