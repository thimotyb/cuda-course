# M9.1.3 Classifier GPU Benchmark Scratchpad

Working notes for what changed to turn M9 exercise 1.3 into a runnable
exercise. Written for whoever is working on 1.4 in parallel (same shared
`.venv`, same `site/chapters/chapter-09.html` file), so conflicts and shared
state are visible up front.

## Heads up for 1.4 before anything else

- **`.venv` now has two new packages**: `scikit-learn` and `xgboost`
  (`.venv/bin/pip install scikit-learn xgboost`, already run). If 1.4 needs
  its own packages, they stack on top of these — no conflicts expected
  (`torch`, `numpy`, `scikit-learn`, `xgboost` only so far).
- **`site/chapters/chapter-09.html` was edited**: only the `1.3` `<article>`
  (line ~36, one long line — this file keeps one `<article>` per line). `1.1`,
  `1.2`, section 2, and everything else were not touched. If 1.4 also edits
  this file, expect to merge/rebase on that one line's neighbors, not
  overlapping content.
- **No non-regression lock exists for M9** (`tests/non-regression/locks/`
  stops at `M8.lock.json`), so neither of us needs to regenerate a lock for
  chapter-09 edits. Still worth running `python3
  scripts/non_regression_guard.py check` before considering things done —
  it also validates structure/numbering across all chapters, not just locked
  ones.
- New example lives at `examples/ch09/classifier_gpu_benchmark/` (script +
  README). If 1.4 also adds an `examples/ch09/<something>/` folder (the
  vLLM Docker demo, referenced in 1.4's own content as
  `examples/ch09/vllm_docker_demo/`), no path collision — different
  subfolder.

## What 1.3 asks and what was built

Original 1.3 was a generic placeholder ("Mini inference benchmark with
PyTorch", no linked example — M9 was a pure skeleton before this work). The
request was to build a concrete Random Forest CPU/GPU comparison, which grew
(by explicit user request, mid-conversation) into a three-model comparison:
Random Forest (scikit-learn), XGBoost, and a small MLP (PyTorch), all on the
same dataset, one script, `--model {rf,xgboost,mlp,all}`.

**Dataset**: scikit-learn's Covertype (`fetch_covtype`) — 581,012 rows, 54
features, 7 classes. Downloads once (~75 MB), cached under
`~/scikit_learn_data`. Chosen over a synthetic `make_classification` set
because the user explicitly wanted something real and "already available"
rather than generated.

**Why three models, not just Random Forest**: the user asked what other
classifiers could actually train on GPU (since Random Forest training
cannot), which led to adding XGBoost (real GPU training via histogram
splits) and an MLP (the textbook PyTorch GPU-native case) as a deliberate
three-way contrast in the same exercise.

## Files

- `examples/ch09/classifier_gpu_benchmark/classifier_gpu_benchmark.py` — the
  benchmark script.
- `examples/ch09/classifier_gpu_benchmark/README.md` — usage + concept
  writeup + measured numbers table.
- `site/chapters/chapter-09.html` — 1.3 rewritten with the same concept
  writeup, a results table, and updated exercise-list bullets.

## How the script is structured

Single file, follows the style already established in `examples/ch08/*.py`
(typed functions, docstrings, `torch.cuda.Event` + `synchronize()` for GPU
timing, plain `time.perf_counter()` otherwise). Key pieces:

- `detect_device()` — returns `cuda:0` or `None`. Unlike ch08's
  `require_cuda()`, this script does **not** stop without a GPU; it just
  skips GPU columns. It has to run meaningfully on a CPU-only machine too.
- `timed(fn, device=None)` — one timing helper used everywhere (sklearn
  calls, xgboost calls, and torch CPU/GPU calls alike). CUDA-event path only
  triggers when `device.type == "cuda"`.
- `load_dataset(...)` — fetch + stratified split + optional random
  subsampling (`--max-train-rows` / `--max-test-rows`, default `200000` /
  `50000`, `-1` = full split) so a full run finishes in a few minutes.
- **Random Forest path** (`extract_forest_tensors` +
  `predict_forest_tensor` + `run_random_forest`): trains with
  `RandomForestClassifier` on CPU (no GPU path in scikit-learn — see "the
  parallelism correction" below for *why*, it's not what you'd guess).
  Inference is reimplemented from scratch: every tree's `feature`,
  `threshold`, `children_left`, `children_right`, and per-leaf class
  distribution are flattened into padded tensors. **The trick that makes
  this simple**: leaf nodes get a self-loop (their children point back to
  themselves), so the traversal can run for a fixed `max_depth` steps with
  no per-sample "already done" mask — a sample that reaches its leaf early
  just keeps landing on the same leaf. Correctness is checked against
  sklearn's own `.predict()` (`match_vs_sklearn`, prints ~100%) before any
  timing is trusted.
- **XGBoost path** (`run_xgboost`): trains twice, `device="cpu"` and
  `device="cuda"`, both with `tree_method="hist"`. The GPU path is wrapped
  in `try/except` in case this environment's XGBoost build doesn't support
  the GPU — **turned out not to be needed**: XGBoost GPU training and
  inference both work fine on this machine's RTX 5060 Ti (Blackwell). One
  real gotcha found while testing: calling `.predict()` with host (CPU)
  numpy data against a GPU-trained booster makes XGBoost silently fall back
  to copying the data to the device on every call (it warns once). That copy
  is left inside the timed GPU-predict measurement on purpose (it's a real,
  relevant cost), the noisy library warning is suppressed
  (`warnings.filterwarnings(..., module="xgboost")`), and a one-line note is
  printed in its place explaining why.
- **MLP path** (`MLP`, `train_mlp`, `evaluate_mlp`, `run_mlp`): plain
  `nn.Sequential` (54→128→64→7), Adam, CrossEntropyLoss, trained once on CPU
  and once on GPU with the same seed. Data is moved to the device once
  before the timed training loop (one-time setup cost, not per-epoch).

## Real numbers (course machine: RTX 5060 Ti, default settings)

`--model all`, `--max-train-rows 200000 --max-test-rows 50000`:

| Model | Phase | CPU | GPU | Speedup |
| --- | --- | --- | --- | --- |
| Random Forest | inference (tensor-walk) | 1073 ms | 116 ms | 9.3x |
| XGBoost | train | 9609 ms | 8268 ms | 1.16x |
| XGBoost | inference | 224 ms | 76 ms | 2.95x |
| MLP | train | 558 ms | 520 ms | 1.07x |
| MLP | inference | 5.6 ms | 8.1 ms | 0.69x — CPU wins |

Notable: MLP *inference* is the one place GPU loses outright here — batch
too small to amortize CUDA launch/sync overhead, same lesson as the M8
PyTorch benchmarks, just showing up on a real classification workload. These
exact numbers are quoted in both the README and the site page; expect
run-to-run variance (a repeat run showed XGBoost CPU train time swing from
~9.4s to ~21s depending on machine load — kept the cleaner run for docs).

## The parallelism correction (important, changed the writeup twice)

First draft of the writeup attributed "Random Forest can't train on GPU" to
tree induction being inherently sequential/branchy, framed as if bagging
itself resisted parallelism compared to boosting. The user pushed back
(citing a Gemini answer) and was right to: **that framing was backwards on
one axis**. Corrected picture, now in all three docs (script docstring,
README, site page):

- **Across trees**: bagging (Random Forest) is embarrassingly parallel —
  trees are independent, scikit-learn already parallelizes this across CPU
  threads (`n_jobs=-1`). Boosting (XGBoost) is strictly sequential across
  trees — tree N+1 needs the residuals of trees 0..N, on any hardware. RF is
  the *more* parallel ensemble structure on this axis, not the less parallel
  one.
- **Inside one tree** (the split-search algorithm): this is what actually
  explains the CPU/GPU split in this exercise. scikit-learn uses exact-greedy
  search (no GPU kernel in that library). XGBoost's `hist` method reframes
  split search as histogram construction/reduction (GPU kernels exist).
  This is a library implementation choice, not a property of Random Forest
  as an algorithm — RAPIDS cuML has a GPU-native `RandomForestClassifier`
  using histogram splits, proving the point. It wasn't used here (RAPIDS
  install risk on this WSL2 + Blackwell setup, discussed and rejected during
  planning — see git history / conversation, not written down elsewhere).

If 1.4 or later content references "why doesn't Random Forest use the GPU",
point at this two-axis explanation instead of a single "trees are branchy"
answer — the single-axis version is what got corrected here.

## Verification run

```bash
.venv/bin/python examples/ch09/classifier_gpu_benchmark/classifier_gpu_benchmark.py --model all
python3 scripts/non_regression_guard.py check   # PASS
```
