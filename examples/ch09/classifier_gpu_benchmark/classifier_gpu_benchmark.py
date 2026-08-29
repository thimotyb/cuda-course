#!/usr/bin/env python3
"""Compare three classifier families on CPU and GPU: Random Forest, XGBoost,
and a small MLP.

The three models have a different relationship with the GPU, and that is the
point of this example:

- Random Forest (scikit-learn) builds independent trees from bootstrap
  samples (bagging), so in principle every tree can be grown at once — and
  scikit-learn's own training already exploits that, parallelizing across
  trees with CPU threads (`n_jobs=-1`). What it does NOT parallelize is the
  work inside one tree: it finds each split with an exact-greedy search
  (scan the real values at a node, try every threshold, keep the best) — a
  small, branchy, data-dependent computation with no CUDA kernel in this
  library. That is a fact about scikit-learn's implementation, not about
  Random Forest as an algorithm (RAPIDS cuML has a GPU-native
  RandomForestClassifier that uses histogram splits instead). Once a forest
  is trained, *predicting* is a different computation: walking a fixed set
  of nodes for many rows at once is regular and batchable, so that part is
  rewritten here as vectorized tensor operations and run on CPU or GPU.
- XGBoost grows gradient boosted trees, and boosting has the opposite
  ensemble structure from bagging: tree N+1 corrects the residual errors of
  trees 0..N, so trees must be built strictly one after another — no amount
  of hardware parallelizes across boosting rounds. What XGBoost does
  accelerate is the work inside each tree: its histogram-based split method
  ("hist") bins feature values and finds splits by building and reducing
  per-feature histograms — a large, regular aggregation, not a sequence of
  small branchy decisions — with real CUDA kernels behind it
  (`device="cuda"`). Each sequential boosting round still gets faster on a
  GPU, even though the rounds themselves cannot overlap.
- A small MLP in PyTorch is the textbook GPU-friendly workload: forward and
  backward passes are dense matrix multiplications and elementwise ops, fully
  regular and data-parallel, so both training and inference benefit from the
  GPU.

Two different axes are at play, and it is easy to collapse them into one:
whether trees in the ensemble can be built independently (bagging: yes;
boosting: no — this has nothing to do with hardware) versus whether the
split-search *inside* one tree is expressed in a GPU-friendly way
(scikit-learn's exact-greedy search: no; XGBoost's histogram method: yes).
Random Forest is the more parallel ensemble structure of the two; it simply
trains on CPU here because of scikit-learn's split-search implementation.

Dataset: scikit-learn's Covertype (`fetch_covtype`), 581,012 rows, 54
features, 7 forest cover-type classes. It downloads once (~75 MB) and is then
cached under `~/scikit_learn_data`.
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


Row = tuple[str, str, str, float, str]


def detect_device() -> Optional[torch.device]:
    """Return cuda:0 if PyTorch can see a CUDA GPU, otherwise None.

    Unlike the ch08 examples, this script does not stop when CUDA is missing:
    it simply prints CPU-only results and skips the GPU columns.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return None


def timed(fn: Callable[[], object], device: Optional[torch.device] = None):
    """Run fn() once and return (result, elapsed_ms).

    Uses CUDA events plus torch.cuda.synchronize() when device is a CUDA
    device, so the timing reflects completed GPU work rather than submission
    time. Otherwise it uses a plain host wall clock, which is also correct for
    CPU-only calls such as scikit-learn or XGBoost training and prediction.
    """
    if device is not None and device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        stop.record()
        torch.cuda.synchronize(device)
        return result, start.elapsed_time(stop)

    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def load_dataset(max_train_rows: int, max_validation_rows: int, seed: int):
    """Load Covertype, split it, and optionally subsample for a faster run.

    The held-out split is used as a validation set: it is never used to fit a
    model, and its accuracy is the common quality metric for all three models.
    max_train_rows / max_validation_rows of -1 keep the full split.
    Subsampling is a plain random draw from the already-stratified split, so
    class proportions stay close to the full dataset.
    """
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    print("Loading Covertype (downloads once, then cached under ~/scikit_learn_data)...")
    X, y = fetch_covtype(return_X_y=True)
    y = y.astype(np.int64) - 1  # sklearn labels are 1..7; shift to 0..6

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    rng = np.random.default_rng(seed)
    if 0 < max_train_rows < len(X_train):
        idx = rng.choice(len(X_train), size=max_train_rows, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if 0 < max_validation_rows < len(X_validation):
        idx = rng.choice(len(X_validation), size=max_validation_rows, replace=False)
        X_validation, y_validation = X_validation[idx], y_validation[idx]

    X_train = X_train.astype(np.float32)
    X_validation = X_validation.astype(np.float32)

    print(
        f"  train: {X_train.shape[0]:,} rows, validation: {X_validation.shape[0]:,} rows, "
        f"features: {X_train.shape[1]}, classes: {len(np.unique(y))}"
    )
    return X_train, X_validation, y_train, y_validation


# ---------------------------------------------------------------------------
# Random Forest: train on CPU with scikit-learn, then run inference as a
# vectorized tensor traversal on CPU or GPU.
# ---------------------------------------------------------------------------


def extract_forest_tensors(clf) -> dict:
    """Flatten a trained RandomForestClassifier into fixed-size tensors.

    Each estimator's tree_ has its own node arrays with a different node
    count, so every array is padded to the largest tree's node count. Leaf
    nodes are given a self-loop: children_left/children_right at a leaf are
    rewritten to point back to that same leaf. That means the traversal below
    can always run for a fixed number of steps (the forest's max_depth) —
    once a sample reaches a leaf, further steps just keep it there, so no
    per-sample "done" mask is needed.
    """
    trees = [est.tree_ for est in clf.estimators_]
    n_estimators = len(trees)
    n_classes = int(clf.n_classes_)
    max_nodes = max(t.node_count for t in trees)

    feature = np.zeros((n_estimators, max_nodes), dtype=np.int64)
    threshold = np.zeros((n_estimators, max_nodes), dtype=np.float32)
    children_left = np.zeros((n_estimators, max_nodes), dtype=np.int64)
    children_right = np.zeros((n_estimators, max_nodes), dtype=np.int64)
    value = np.zeros((n_estimators, max_nodes, n_classes), dtype=np.float32)

    for i, tree in enumerate(trees):
        n = tree.node_count
        feature[i, :n] = tree.feature
        threshold[i, :n] = tree.threshold

        left = tree.children_left.copy()
        right = tree.children_right.copy()
        leaf_positions = np.nonzero(left == -1)[0]  # sklearn marks leaves with -1
        left[leaf_positions] = leaf_positions
        right[leaf_positions] = leaf_positions
        children_left[i, :n] = left
        children_right[i, :n] = right

        counts = tree.value[:, 0, :]  # (n, n_classes) unnormalized class counts
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        value[i, :n, :] = counts / row_sums

    return {
        "feature": torch.from_numpy(feature),
        "threshold": torch.from_numpy(threshold),
        "children_left": torch.from_numpy(children_left),
        "children_right": torch.from_numpy(children_right),
        "value": torch.from_numpy(value),
        "n_classes": n_classes,
    }


def predict_forest_tensor(
    X_np: np.ndarray, forest: dict, device: torch.device, max_depth: int
) -> torch.Tensor:
    """Predict class labels by walking every tree for every row at once.

    node_idx has shape (n_estimators, n_samples): the current node index for
    every (tree, sample) pair. Each step gathers the feature and threshold of
    the current node, compares it against the matching input column, and
    moves to the left or right child — all as batched tensor ops instead of a
    Python-level recursion per sample. Because trees were trained with a
    bounded max_depth, max_depth steps are always enough to reach a leaf for
    every row (the self-loop keeps rows that arrive early where they are).
    """
    n_estimators, _ = forest["feature"].shape
    n_classes = forest["n_classes"]
    X = torch.from_numpy(X_np).to(device=device, dtype=torch.float32)
    n_samples = X.shape[0]

    feature = forest["feature"]
    threshold = forest["threshold"]
    children_left = forest["children_left"]
    children_right = forest["children_right"]
    value = forest["value"]

    node_idx = torch.zeros((n_estimators, n_samples), dtype=torch.long, device=device)
    sample_rows = torch.arange(n_samples, device=device).unsqueeze(0).expand(n_estimators, n_samples)

    for _ in range(max_depth):
        feat = torch.gather(feature, 1, node_idx)
        thresh = torch.gather(threshold, 1, node_idx)
        x_vals = X[sample_rows, feat]
        go_left = x_vals <= thresh
        left_child = torch.gather(children_left, 1, node_idx)
        right_child = torch.gather(children_right, 1, node_idx)
        node_idx = torch.where(go_left, left_child, right_child)

    node_idx_expanded = node_idx.unsqueeze(-1).expand(-1, -1, n_classes)
    leaf_probs = torch.gather(value, 1, node_idx_expanded)  # (n_estimators, n_samples, n_classes)
    avg_probs = leaf_probs.mean(dim=0)  # soft-vote average across trees, same as sklearn's predict()
    return avg_probs.argmax(dim=1)


def run_random_forest(X_train, y_train, X_validation, y_validation, args, device) -> list[Row]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rows: list[Row] = []

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=args.seed,
    )
    _, train_ms = timed(lambda: clf.fit(X_train, y_train))
    print(
        f"  sklearn RandomForestClassifier.fit (CPU only, no GPU path): {train_ms:9.1f} ms  "
        f"[n_estimators={args.n_estimators}, max_depth={args.max_depth}]"
    )
    rows.append(("Random Forest", "train", "cpu", train_ms, "scikit-learn, no GPU path"))

    sk_preds, sk_predict_ms = timed(lambda: clf.predict(X_validation))
    sk_acc = accuracy_score(y_validation, sk_preds)
    print(f"  sklearn .predict() (CPU, native, validation reference):      {sk_predict_ms:9.1f} ms  accuracy={sk_acc:.4f}")

    forest = extract_forest_tensors(clf)

    for dev in [d for d in (torch.device("cpu"), device) if d is not None]:
        forest_dev = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in forest.items()}
        preds, ms = timed(lambda: predict_forest_tensor(X_validation, forest_dev, dev, args.max_depth), dev)
        preds_np = preds.cpu().numpy()
        match_rate = float((preds_np == sk_preds).mean())
        acc = accuracy_score(y_validation, preds_np)
        rows_per_s = len(X_validation) / (ms / 1000.0)
        note = f"accuracy={acc:.4f} match_vs_sklearn={match_rate:.4%} throughput={rows_per_s:,.0f} rows/s"
        print(f"  torch tensor-walk predict on {dev.type:<4}:                   {ms:9.1f} ms  {note}")
        rows.append(("Random Forest", "inference", dev.type, ms, note))

    return rows


# ---------------------------------------------------------------------------
# XGBoost: gradient boosted trees, hist split method — can train on GPU.
# ---------------------------------------------------------------------------


def run_xgboost(X_train, y_train, X_validation, y_validation, args, device) -> list[Row]:
    import warnings

    from sklearn.metrics import accuracy_score

    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost is not installed (pip install xgboost). Skipping this model.")
        return []

    # predict() below passes plain numpy (host) arrays. Against a GPU-trained
    # booster, XGBoost silently copies that data to the device on every call
    # instead of raising an error; it only warns once. That copy is a real,
    # relevant cost — it is the same "hidden transfer overhead" lesson as
    # elsewhere in this module — so we keep it in the timing but replace the
    # noisy library warning with our own one-line note below.
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

    rows: list[Row] = []

    def make_clf(device_name: str):
        return xgb.XGBClassifier(
            n_estimators=args.xgb_estimators,
            max_depth=args.xgb_max_depth,
            tree_method="hist",
            device=device_name,
            eval_metric="mlogloss",
            random_state=args.seed,
        )

    cpu_clf = make_clf("cpu")
    _, cpu_train_ms = timed(lambda: cpu_clf.fit(X_train, y_train))
    cpu_preds, cpu_predict_ms = timed(lambda: cpu_clf.predict(X_validation))
    cpu_acc = accuracy_score(y_validation, cpu_preds)
    print(f"  XGBoost train, hist method (CPU):    {cpu_train_ms:9.1f} ms")
    print(f"  XGBoost predict (CPU):               {cpu_predict_ms:9.1f} ms  accuracy={cpu_acc:.4f}")
    rows.append(("XGBoost", "train", "cpu", cpu_train_ms, "hist tree method"))
    rows.append(
        (
            "XGBoost",
            "inference",
            "cpu",
            cpu_predict_ms,
            f"accuracy={cpu_acc:.4f} throughput={len(X_validation) / (cpu_predict_ms / 1000.0):,.0f} rows/s",
        )
    )

    if device is None:
        print("  CUDA not available in this environment; skipping XGBoost GPU columns.")
        return rows

    try:
        gpu_clf = make_clf("cuda")
        _, gpu_train_ms = timed(lambda: gpu_clf.fit(X_train, y_train))
        print("  Note: predict() below passes host (CPU) numpy data to a GPU-trained")
        print("  booster, so XGBoost copies it to the device on every call — that copy")
        print("  is included in the GPU predict time below.")
        gpu_preds, gpu_predict_ms = timed(lambda: gpu_clf.predict(X_validation))
    except Exception as exc:  # noqa: BLE001 - genuinely want to catch any backend failure here
        print(f"  XGBoost GPU training/inference is not available in this environment: {exc}")
        print("  Skipping XGBoost GPU columns; the CPU results above still stand.")
        return rows

    gpu_acc = accuracy_score(y_validation, gpu_preds)
    print(
        f"  XGBoost train, hist method (GPU):    {gpu_train_ms:9.1f} ms  "
        f"speedup={cpu_train_ms / gpu_train_ms:.2f}x"
    )
    print(
        f"  XGBoost predict (GPU):               {gpu_predict_ms:9.1f} ms  accuracy={gpu_acc:.4f}  "
        f"speedup={cpu_predict_ms / gpu_predict_ms:.2f}x"
    )
    rows.append(("XGBoost", "train", "cuda", gpu_train_ms, "hist tree method"))
    rows.append(
        (
            "XGBoost",
            "inference",
            "cuda",
            gpu_predict_ms,
            f"accuracy={gpu_acc:.4f} throughput={len(X_validation) / (gpu_predict_ms / 1000.0):,.0f} rows/s",
        )
    )
    return rows


# ---------------------------------------------------------------------------
# MLP: fully connected classifier in pure PyTorch — trains and infers on GPU.
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(device, X, y, n_features, n_classes, epochs, batch_size, seed):
    """Train a fresh MLP on `device`. X and y are already on that device."""
    torch.manual_seed(seed)
    model = MLP(n_features, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]

    def loop():
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                out = model(X[idx])
                loss = loss_fn(out, y[idx])
                loss.backward()
                optimizer.step()
        return model

    trained_model, train_ms = timed(loop, device)
    return trained_model, train_ms


def evaluate_mlp(model, device, X, y_np, batch_size):
    model.eval()

    def run():
        chunks = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                out = model(X[start : start + batch_size])
                chunks.append(out.argmax(dim=1))
        return torch.cat(chunks)

    preds, infer_ms = timed(run, device)
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_np, preds.cpu().numpy())
    return acc, infer_ms


def run_mlp(X_train, y_train, X_validation, y_validation, args, device) -> list[Row]:
    n_features = X_train.shape[1]
    n_classes = int(y_train.max()) + 1

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_validation_t = torch.from_numpy(X_validation).float()

    rows: list[Row] = []
    train_ms_by_device = {}
    for dev in [d for d in (torch.device("cpu"), device) if d is not None]:
        # Moving the whole training set to the device once, before the timed
        # training loop, mirrors real usage: data placement is a one-time
        # setup cost, not something repeated every epoch.
        X_dev = X_train_t.to(dev)
        y_dev = y_train_t.to(dev)
        model, train_ms = train_mlp(dev, X_dev, y_dev, n_features, n_classes, args.epochs, args.mlp_batch_size, args.seed)
        acc, infer_ms = evaluate_mlp(model, dev, X_validation_t.to(dev), y_validation, args.mlp_batch_size)
        train_ms_by_device[dev.type] = train_ms
        print(f"  MLP train on {dev.type:<4} ({args.epochs} epochs):            {train_ms:9.1f} ms")
        print(f"  MLP inference on {dev.type:<4}:                         {infer_ms:9.1f} ms  accuracy={acc:.4f}")
        rows.append(("MLP", "train", dev.type, train_ms, f"{args.epochs} epochs"))
        rows.append(
            (
                "MLP",
                "inference",
                dev.type,
                infer_ms,
                f"accuracy={acc:.4f} throughput={len(X_validation) / (infer_ms / 1000.0):,.0f} rows/s",
            )
        )

    if "cpu" in train_ms_by_device and "cuda" in train_ms_by_device:
        speedup = train_ms_by_device["cpu"] / train_ms_by_device["cuda"]
        print(f"  MLP training speedup, GPU over CPU: {speedup:.2f}x")

    return rows


# ---------------------------------------------------------------------------


def print_summary(rows: list[Row]) -> None:
    print("\n=== Summary ===")
    header = f"{'Model':<14}{'Phase':<11}{'Device':<7}{'Time (ms)':>12}   Notes"
    print(header)
    print("-" * len(header))
    for model, phase, dev, ms, note in rows:
        print(f"{model:<14}{phase:<11}{dev:<7}{ms:12.1f}   {note}")


def print_accuracy_summary(rows: list[Row]) -> None:
    """Print validation accuracy in a compact model-comparison table."""
    print("\n=== Validation accuracy comparison ===")
    print(f"{'Model':<14}{'Device':<8}{'Accuracy':>10}")
    print("-" * 34)
    for model, phase, dev, _ms, note in rows:
        if phase != "inference":
            continue
        match = re.search(r"accuracy=([0-9.]+)", note)
        if match:
            print(f"{model:<14}{dev:<8}{float(match.group(1)):10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Random Forest, XGBoost, and an MLP classifier on CPU vs GPU."
    )
    parser.add_argument("--model", choices=["rf", "xgboost", "mlp", "all"], default="all")
    parser.add_argument("--n-estimators", type=int, default=200, help="Random Forest tree count.")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="Random Forest max depth. Bounds the tensor traversal to a fixed step count.",
    )
    parser.add_argument("--xgb-estimators", type=int, default=200, help="XGBoost boosting rounds.")
    parser.add_argument("--xgb-max-depth", type=int, default=8, help="XGBoost tree max depth.")
    parser.add_argument("--epochs", type=int, default=5, help="MLP training epochs.")
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument(
        "--max-train-rows", type=int, default=200_000, help="-1 to use the full training split."
    )
    parser.add_argument(
        "--max-validation-rows",
        type=int,
        default=50_000,
        help="validation rows used for timing and accuracy; -1 keeps the full validation split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = detect_device()
    if device is not None:
        print(f"CUDA available: True ({torch.cuda.get_device_name(device)})")
    else:
        print("CUDA available: False — running CPU-only, GPU columns will be skipped.")

    X_train, X_validation, y_train, y_validation = load_dataset(
        args.max_train_rows, args.max_validation_rows, args.seed
    )

    all_rows: list[Row] = []
    if args.model in ("rf", "all"):
        print("\n=== Random Forest (scikit-learn train, PyTorch tensor-walk inference) ===")
        all_rows += run_random_forest(X_train, y_train, X_validation, y_validation, args, device)
    if args.model in ("xgboost", "all"):
        print("\n=== XGBoost (gradient boosted trees, hist split method) ===")
        all_rows += run_xgboost(X_train, y_train, X_validation, y_validation, args, device)
    if args.model in ("mlp", "all"):
        print("\n=== MLP (pure PyTorch) ===")
        all_rows += run_mlp(X_train, y_train, X_validation, y_validation, args, device)

    if args.model == "all":
        print_summary(all_rows)
        print_accuracy_summary(all_rows)


if __name__ == "__main__":
    main()
