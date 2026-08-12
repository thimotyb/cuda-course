# Book/Module Mapping Memo

## 2026-08-05 - Book chapter 3 vs course M3

Decision: keep course M3 as "GPU Compute Architecture".

Rationale:

- The official course syllabus defines M3 as GPU compute architecture: modern GPU structure, SMs, CUDA cores, block scheduling, synchronization, transparent scalability, warps, divergence, warp scheduling, latency tolerance, occupancy, and resource partitioning.
- These topics match the book chapter 4, "Compute architecture and scheduling", not book chapter 3.
- Book chapter 3 extends the programming-model material from chapter 2 with multidimensional grids/data, image processing examples, blur processing, and matrix multiplication setup.
- For the current course flow, book chapter 3 is a useful optional bridge after M2, but it is not required to satisfy the syllabus scope for M3.

Follow-up candidate if there is time:

- Add an optional M2 appendix or short extra lesson on multidimensional grids and image blur.
- Reuse the book chapter 3 material as a bridge from 1D vector kernels to 2D data kernels.
- Coordinate with M4 before adding matrix multiplication material, because M4 already covers memory locality, tiling, and matrix multiplication from the memory-architecture angle.
