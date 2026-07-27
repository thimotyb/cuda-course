# Course Structure

- Course name: GPUs and CUDA for Parallel Computing
- Audience: Developers, ML engineers, technical students, and architects with basic C/C++ or Python familiarity
- Language: English primary; Italian via on-page Google Translate switch
- Output format: Static interactive course website for Netlify
- Theme: Copper Current

## Modules

| Module | Title | Output Artifact | Primary Sources | Notes |
| --- | --- | --- | --- | --- |
| M1 | Introduction to Heterogeneous Parallel Computing | `site/chapters/chapter-01.html` | `cuda-course-contents.md`, `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf`, NVIDIA CUDA Programming Guide | Foundations: serial vs parallel vs heterogeneous systems |
| M2 | CUDA Programming Model | `site/chapters/chapter-02.html` | `cuda-course-contents.md`, `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf` | Kernels, runtime API, memory transfers, vector addition |
| M3 | GPU Compute Architecture | `site/chapters/chapter-03.html` | `cuda-course-contents.md`, `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf`, How to Think About GPUs | SMs, warps, scheduling, occupancy |
| M4 | Memory Architecture and Data Locality | `site/chapters/chapter-04.html` | `cuda-course-contents.md`, `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf` | Memory hierarchy, tiling, matrix multiplication |
| M5 | Performance Measurement | `site/chapters/chapter-05.html` | `cuda-course-contents.md`, `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf` | Timing, bottlenecks, kernel vs transfer analysis |
| M6 | Memory Optimization | `site/chapters/chapter-06.html` | `cuda-course-contents.md`, `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf`, CUDA Toolkit resources | Pinned memory, async copies, streams, memory-space choices |
| M7 | CUDA Configuration Optimization | `site/chapters/chapter-07.html` | `cuda-course-contents.md`, `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf` | Block sizing, occupancy, tuning strategy |
| M8 | CUDA and PyTorch | `site/chapters/chapter-08.html` | `cuda-course-contents.md`, PyTorch CUDA docs, CUDA toolkit/docs, accelerated-computing-hub | Applied GPU usage from Python and tensor workflows |

## Source Inventory

| Source | Type | Used In Modules | Notes |
| --- | --- | --- | --- |
| `cuda-course-contents.md` | local syllabus | `M1-M8` | Canonical module split and scope |
| `resources/ProgrammingMassivelyParallel-5th-ed-9780443439018.pdf` | local PDF | `M1-M7` | Primary technical source for CUDA fundamentals and optimization concepts |
| `https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html` | official docs | `M1, M2, M6` | High-level framing and platform orientation |
| `https://developer.nvidia.com/cuda/toolkit` | official portal | `M6, M8` | Toolkit, libraries, tutorials, Nsight, training |
| `https://jax-ml.github.io/scaling-book/gpus/` | technical essay | `M1, M3, M7` | Modern mental model for SMs, bandwidth, and throughput reasoning |
| `https://github.com/thimotyb/accelerated-computing-hub` | curated repository | `M6, M8` | Tutorials and exercises for CUDA C++, Python, and deployment |
| `https://www.manning.com/books/cuda-for-deep-learning` | book landing page | `M8` | Supplemental orientation for deep learning workflows |
| `https://docs.pytorch.org/docs/stable/cuda.html` | official docs | `M8` | `torch.cuda` reference and workflow anchors |

## Mapping Rules

- Keep one module page per module under `site/chapters/`.
- Keep module IDs stable from `M1` to `M8`.
- Use section numbering that follows the course, not the underlying source numbering.
- Keep the left outline navigation aligned 1:1 with page `h2` and `h3`.
- Author content in English and expose Italian through the page language switch.
