# Classifier GPU Benchmark: Random Forest, XGBoost, MLP

This example trains the same classification task with three different model
families and measures CPU versus GPU on each of them. The point is not "GPU is
faster" — it is that the three families have a genuinely different
relationship with the GPU:

- **Random Forest** (`scikit-learn`) builds independent trees from bootstrap
  samples — bagging — so in principle every tree can be grown at once; that
  is exactly why scikit-learn already parallelizes training across trees on
  CPU threads (`n_jobs=-1`). What it does *not* parallelize is the work
  inside one tree: scikit-learn finds each split with an exact-greedy
  search (scan the real values at a node, try every threshold, keep the
  best) — small, branchy, data-dependent, and with no CUDA kernel behind it
  in this library. That is a fact about scikit-learn's implementation, not
  about Random Forest as an algorithm (RAPIDS cuML has a GPU-native
  `RandomForestClassifier` that uses histogram splits instead — see "Two
  kinds of parallelism" below). Once the forest is trained, *predicting* is
  a different computation — walking a fixed set of nodes for many rows at
  once — regular and batchable, so that part is rewritten here as vectorized
  PyTorch tensor ops and benchmarked on CPU and GPU.
- **XGBoost** grows gradient boosted trees, and boosting has the opposite
  ensemble structure from bagging: tree N+1 corrects the residual errors of
  trees 0..N, so trees must be built strictly one after another — no amount
  of hardware parallelizes across boosting rounds. What XGBoost *does*
  accelerate is the work inside each tree: its histogram-based split method
  (`tree_method="hist"`) bins feature values and finds splits by building
  and reducing per-feature histograms — a large, regular aggregation instead
  of many small branchy decisions — with real CUDA kernels behind it
  (`device="cuda"`). Each sequential boosting round still gets faster on a
  GPU, even though the rounds themselves cannot overlap. XGBoost is not
  built on PyTorch — it has its own CUDA kernels — but it is a realistic,
  widely used example of a GPU-trainable tree ensemble.
- **MLP** (pure PyTorch) is the textbook GPU-friendly workload: forward and
  backward passes are dense matrix multiplications and elementwise ops, fully
  regular and data-parallel, so both training and inference can use the GPU.

### Two kinds of parallelism, not one

It is easy to collapse two separate questions into one. First: can the trees
in the ensemble be built independently of each other? Bagging (Random
Forest) says yes — trees are embarrassingly parallel across the ensemble.
Boosting (XGBoost) says no — trees are strictly sequential across the
ensemble, on any hardware. On *this* axis Random Forest is the more
parallel structure, not the less parallel one.

Second, separate question: how is the best split found while growing *one*
tree? This is where the GPU story in this example actually comes from —
exact-greedy search (scikit-learn) has no GPU kernel here; histogram-based
search (XGBoost's `hist` method, and RAPIDS cuML's GPU Random Forest) does.
So "our Random Forest trains on CPU only" is a statement about which
split-search algorithm scikit-learn implements, not about bagging being
harder to parallelize than boosting.

Dataset: scikit-learn's **Covertype** (`sklearn.datasets.fetch_covtype`) —
581,012 rows, 54 features, 7 forest cover-type classes. It downloads once
(~75 MB) the first time the script runs and is cached under
`~/scikit_learn_data` after that.

## Requirements

- Python 3
- PyTorch with CUDA support
- NumPy
- scikit-learn
- xgboost
- An NVIDIA GPU visible to PyTorch (optional — the script runs CPU-only and
  skips GPU columns if CUDA is not available)

From the repository root, extend the shared virtual environment:

```bash
cd /home/thimoty/git/cuda-course
python3 -m venv .venv  # if it does not exist yet
.venv/bin/python -m pip install torch numpy --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install scikit-learn xgboost
```

If `.venv` already exists, just add the two new packages:

```bash
cd /home/thimoty/git/cuda-course
.venv/bin/pip install scikit-learn xgboost
```

## Run

From the repository root, run all three models and print a summary table:

```bash
.venv/bin/python examples/ch09/classifier_gpu_benchmark/classifier_gpu_benchmark.py --model all
```

Or run one model at a time:

```bash
.venv/bin/python examples/ch09/classifier_gpu_benchmark/classifier_gpu_benchmark.py --model rf
.venv/bin/python examples/ch09/classifier_gpu_benchmark/classifier_gpu_benchmark.py --model xgboost
.venv/bin/python examples/ch09/classifier_gpu_benchmark/classifier_gpu_benchmark.py --model mlp
```

Useful flags (defaults are sized to finish in a few minutes on a course
machine):

- `--max-train-rows`, `--max-test-rows` (default `200000` / `50000`, `-1` for
  the full split) — control dataset size and runtime.
- `--n-estimators`, `--max-depth` — Random Forest tree count and depth. Depth
  is capped deliberately: it bounds the tensor-walk traversal to a fixed
  number of steps (see "How the Random Forest inference works" below).
- `--xgb-estimators`, `--xgb-max-depth` — XGBoost boosting rounds and depth.
- `--epochs`, `--mlp-batch-size` — MLP training epochs and batch size.

## How the Random Forest inference works

A trained Random Forest is exported into fixed-size tensors: for every tree,
`feature`, `threshold`, `children_left`, `children_right`, and the per-leaf
class distribution (`value`), padded to the largest tree's node count. Leaf
nodes get a **self-loop** — their `children_left`/`children_right` are
rewritten to point back to themselves — so a fixed number of traversal steps
(the forest's `max_depth`) is always enough: a sample that reaches its leaf
early just keeps "arriving" at the same leaf for the remaining steps, with no
per-sample mask needed. Each step gathers the current node's feature and
threshold for every (tree, sample) pair at once, compares it against the
input, and moves to the left or right child as a single batched tensor
operation — CPU and GPU run the exact same code, just on a different device.

## Expected behavior

The program prints, for each requested model: a training-time line, an
inference-time line with accuracy, and (where a GPU is available) the same
numbers again on `cuda:0` with a speedup ratio. `--model rf` also prints a
`match_vs_sklearn` rate — the tensor-walk predictions should match
scikit-learn's own `.predict()` almost exactly, since both average the same
per-tree class probabilities. `--model all` finishes with one summary table
across all three models.

### Numbers observed on the course reference machine

RTX 5060 Ti, default settings (`--max-train-rows 200000 --max-test-rows
50000`), `--model all`:

| Model | Phase | Device | Time | Notes |
| --- | --- | --- | --- | --- |
| Random Forest | train | CPU | 7022 ms | scikit-learn, no GPU path |
| Random Forest | inference | CPU | 1073 ms | tensor-walk, 46.6k rows/s |
| Random Forest | inference | GPU | 116 ms | tensor-walk, 431.7k rows/s, **9.3x** |
| XGBoost | train | CPU | 9609 ms | hist method |
| XGBoost | train | GPU | 8268 ms | hist method, **1.16x** |
| XGBoost | inference | CPU | 224 ms | 223.5k rows/s |
| XGBoost | inference | GPU | 76 ms | 660.1k rows/s, **2.95x** |
| MLP | train | CPU | 558 ms | 5 epochs |
| MLP | train | GPU | 520 ms | 5 epochs, **1.07x** |
| MLP | inference | CPU | 5.6 ms | 8.9M rows/s |
| MLP | inference | GPU | 8.1 ms | 6.2M rows/s, **0.69x (CPU wins)** |

Three takeaways worth noticing, not just the headline "GPU vs CPU" numbers:

- Random Forest inference is the clearest GPU win here (~9x): it is exactly
  the kind of large, regular, batched workload the tensor-walk was designed
  to expose.
- XGBoost's GPU *training* speedup is modest at this dataset size (`1.16x`) —
  histogram construction is GPU-friendly, but at 200k rows and 200 shallow
  trees the CPU `hist` method is already efficient; the crossover point moves
  with dataset size, tree count, and depth. Try `--max-train-rows -1` (the
  full ~464k-row training split) to see it shift.
- The MLP is the one place the GPU is actually *slower* than the CPU, for
  **inference**. The model and batch are simply too small to amortize CUDA
  launch and synchronization overhead — the same lesson as the M8 PyTorch
  benchmarks (`examples/ch08/pytorch_cuda_tensors`), applied to a real
  classification workload instead of a matrix multiplication.

Run it yourself and expect different absolute numbers — the shape of the
comparison (which model benefits from the GPU, and where) is the point, not
these exact milliseconds.
