## Array Add On Device

This M6 example compares two kernels that do the same 2D array addition but map thread indices to row-major memory differently.

- `add_v1`: adjacent lanes vary the row index `i`, so their flattened addresses are separated by `M` elements.
- `add_v2`: adjacent lanes vary the column index `j`, so their flattened addresses are adjacent.

The point is memory access, not arithmetic. The two kernels perform the same addition, but `add_v1` creates a strided/uncoalesced access pattern while `add_v2` creates a coalesced one.

## 1. Compile and run

From the repository root:

```bash
cd examples/ch06/array_add_on_device
nvcc array_add_on_device.cu -O3 -lineinfo -o array_add_on_device
./array_add_on_device 4096 4096 16 16
```

Arguments are:

```text
./array_add_on_device <rows> <cols> <threads_y> <threads_x>
```

Use dimensions divisible by the block dimensions for this simple teaching version.

## 2. Thread mapping

The only difference between the two kernels is how `i` and `j` are computed:

```C
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;
```

Then the index is computed by this line:
```C
int idx = IDX(i, j, M);
```
which is equivalent to:
```C
int idx = i * M + j
```

Because CUDA warps are formed from consecutive linear thread indices and `threadIdx.x` varies fastest, the first mapping sends adjacent lanes to addresses `M` elements apart. The second mapping sends adjacent lanes to adjacent addresses.

## 3. Profile

Use Nsight Compute to compare memory behavior:

```bash
ncu --set full --kernel-name regex:'^add_v1' \
  --force-overwrite -o array-add-v1-ncu ./array_add_on_device 4096 4096 16 16
ncu --set full --kernel-name regex:'^add_v2' \
  --force-overwrite -o array-add-v2-ncu ./array_add_on_device 4096 4096 16 16
```

Inspect memory throughput, requested versus actual global-memory traffic, and load/store efficiency. `add_v2` should be the better access pattern because it lets the hardware coalesce adjacent lane accesses.
