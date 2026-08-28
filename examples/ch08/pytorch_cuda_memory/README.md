# PyTorch CUDA Memory Tools

This example demonstrates the most useful PyTorch CUDA memory tools for a first GPU workflow.

The goal is to make the CUDA caching allocator visible. PyTorch may reserve GPU memory for reuse even after a tensor is deleted, so `memory_allocated` and `memory_reserved` answer different questions.

Reserved CUDA memory is not pinned memory. In this example, reserved memory means GPU device memory held by PyTorch's CUDA caching allocator. Pinned memory means page-locked CPU host memory used to make host-device transfers more efficient.

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
.venv/bin/python examples/ch08/pytorch_cuda_memory/pytorch_cuda_memory.py
```

Optional tensor size:

```bash
.venv/bin/python examples/ch08/pytorch_cuda_memory/pytorch_cuda_memory.py --size 4096
```

Print a compact `memory_summary()` at the end:

```bash
.venv/bin/python examples/ch08/pytorch_cuda_memory/pytorch_cuda_memory.py --summary
```

## Expected behavior

The program prints memory state:

- before any large allocation;
- after allocating CUDA tensors;
- after running a matrix multiplication;
- after deleting one tensor;
- after deleting all user-created tensors;
- after calling `torch.cuda.empty_cache()`;
- after resetting and reporting peak memory statistics.

The important observation is that allocated memory follows live tensors more closely, while reserved memory reflects memory held by PyTorch's CUDA caching allocator for possible reuse. After library-backed operations such as matrix multiplication, some internal allocations or workspaces may also remain visible.
