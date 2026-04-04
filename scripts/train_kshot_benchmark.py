from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tabdpt import TabDPTRegressor
import torch
from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings("ignore")

REFERENCE_CSV = Path("data/proteingym/reference_files/DMS_Substitutions.csv")
VARIANT_TABLE_DIR = Path("data/proteingym/variant_tables/baseline")
FOLD_DIR = Path("data/proteingym/cv_folds_singles_substitutions")
OUTER_FOLD_COL = "fold_random_5"
META_COLS = {
    "file_id",
    "UniProt_ID",
    "mutant",
    "L_res",
    "DMS_score",
    "DMS_score_bin",
    OUTER_FOLD_COL,
}
TABDPT_DEFAULT_MAX_FEATURES = 100
K_SWEEP = [8, 16, 32, 64, 128, 256, 512]  # -1 sentinel appended per-assay for "all"
OUTER_FOLDS = [0, 1, 2, 3, 4]
ALPHA_GRID = np.logspace(-4, 4, num=9)

# tabdpt checkpoints were saved on CUDA. Patch torch.load so:
#   1. tensors are mapped to CPU on load
#   2. the device field inside the checkpoint config is rewritten to 'cpu'
#      (tabdpt reads config['env']['device'] and calls model.to(...) with it,
#      ignoring the device= argument passed to TabDPTRegressor/Classifier)
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    result = _orig_torch_load(*args, **kwargs)
    if isinstance(result, dict):
        cfg = result.get("cfg", {})
        env = cfg.get("env", {}) if isinstance(cfg, dict) else {}
        if isinstance(env.get("device"), str) and env["device"].startswith("cuda"):
            env["device"] = "cpu"
    return result


torch.load = _patched_torch_load


def load_reference_metadata(reference_csv: Path = REFERENCE_CSV) -> pd.DataFrame:
    """Load ProteinGym assay metadata and key it by file_id."""
    ref = pd.read_csv(reference_csv)
    ref["file_id"] = ref["DMS_filename"].map(lambda x: Path(x).stem)
    return ref.set_index("file_id", drop=False)


def load_assay_data(
    file_id: str,
    variant_table_dir: Path = VARIANT_TABLE_DIR,
    fold_dir: Path = FOLD_DIR,
) -> pd.DataFrame:
    """Load one featurized assay and merge it with official random fold assignments."""
    assay_path = variant_table_dir / f"{file_id}.parquet"
    fold_path = fold_dir / f"{file_id}.csv"

    assay_df = pd.read_parquet(assay_path)
    fold_df = pd.read_csv(fold_path, usecols=["mutant", OUTER_FOLD_COL])
    merged = assay_df.merge(fold_df, on="mutant", how="inner", validate="one_to_one")

    if merged.empty:
        raise ValueError(f"{file_id}: merged assay/fold table is empty")
    if OUTER_FOLD_COL not in merged.columns:
        raise ValueError(f"{file_id}: missing required fold column {OUTER_FOLD_COL!r}")

    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns for a merged assay dataframe."""
    return [c for c in df.columns if c not in META_COLS]


def sample_k_from_training_folds(
    train_df: pd.DataFrame,
    k: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample k rows uniformly at random from the outer-training pool."""
    if len(train_df) < k:
        raise ValueError(
            f"Not enough samples in training pool: need {k}, have {len(train_df)}"
        )
    idx = rng.choice(train_df.index.to_numpy(), size=k, replace=False)
    return train_df.loc[idx].reset_index(drop=True)


def prepare_outer_fold_data(
    assay_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int,
    outer_fold: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Prepare outer train/test splits, few-shot sampling, arrays, and scaling.

    k=-1 means use the full outer-training pool without subsampling.
    """
    train_df = assay_df.loc[assay_df[OUTER_FOLD_COL] != outer_fold].copy()
    test_df = assay_df.loc[assay_df[OUTER_FOLD_COL] == outer_fold].copy()
    sampled_train_df = (
        train_df.reset_index(drop=True)
        if k == -1
        else sample_k_from_training_folds(train_df, k=k, rng=rng)
    )

    X_train = sampled_train_df[feature_cols].to_numpy()
    y_train = sampled_train_df["DMS_score"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["DMS_score"].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "sampled_train_df": sampled_train_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
    }


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman with a finite fallback for tiny or degenerate splits."""
    stat = spearmanr(y_true, y_pred).statistic
    return 0.0 if np.isnan(stat) else float(stat)


def select_ridge_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: np.ndarray,
    inner_n_splits: int,
    random_state: int = 0,
) -> tuple[float, pd.DataFrame]:
    """Choose ridge alpha by inner CV using Spearman on validation folds."""
    cv = KFold(n_splits=inner_n_splits, shuffle=True, random_state=random_state)
    rows = []

    for alpha in alpha_grid:
        fold_scores = []
        for inner_train_idx, inner_val_idx in cv.split(X_train):
            X_inner_train = X_train[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            X_inner_val = X_train[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]

            scaler = StandardScaler()
            X_inner_train_scaled = scaler.fit_transform(X_inner_train)
            X_inner_val_scaled = scaler.transform(X_inner_val)

            model = Ridge(alpha=float(alpha))
            model.fit(X_inner_train_scaled, y_inner_train)
            y_inner_pred = model.predict(X_inner_val_scaled)
            fold_scores.append(safe_spearman(y_inner_val, y_inner_pred))

        rows.append(
            {
                "alpha": float(alpha),
                "mean_spearman": float(np.mean(fold_scores)),
                "std_spearman": float(np.std(fold_scores)),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=["mean_spearman", "alpha"], ascending=[False, True]
    )
    best_alpha = float(results_df.iloc[0]["alpha"])
    return best_alpha, results_df.reset_index(drop=True)


def run_outer_fold_ridge(
    assay_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int,
    outer_fold: int,
    rng: np.random.Generator,
    alpha_grid: np.ndarray,
    random_state: int = 0,
) -> dict[str, object]:
    """Run one ridge outer-fold evaluation on a single assay."""
    prepared = prepare_outer_fold_data(
        assay_df,
        feature_cols,
        k=k,
        outer_fold=outer_fold,
        rng=rng,
    )

    # RidgeCV uses efficient LOO (MSE) instead of nested Spearman CV — much faster
    model = RidgeCV(alphas=alpha_grid)
    model.fit(prepared["X_train_scaled"], prepared["y_train"])
    best_alpha = float(model.alpha_)
    y_pred = model.predict(prepared["X_test_scaled"])

    return {
        "train_df": prepared["train_df"],
        "test_df": prepared["test_df"],
        "sampled_train_df": prepared["sampled_train_df"],
        "X_train": prepared["X_train"],
        "y_train": prepared["y_train"],
        "X_test": prepared["X_test"],
        "y_test": prepared["y_test"],
        "X_train_scaled": prepared["X_train_scaled"],
        "X_test_scaled": prepared["X_test_scaled"],
        "y_pred": y_pred,
        "best_alpha": best_alpha,
        "spearman": safe_spearman(prepared["y_test"], y_pred),
        "mse": float(np.mean((prepared["y_test"] - y_pred) ** 2)),
    }


def run_outer_fold_lightgbm(
    assay_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int,
    outer_fold: int,
    rng: np.random.Generator,
    random_state: int = 0,
) -> dict[str, object]:
    """Run one LightGBM outer-fold evaluation on a single assay."""
    prepared = prepare_outer_fold_data(
        assay_df,
        feature_cols,
        k=k,
        outer_fold=outer_fold,
        rng=rng,
    )

    model = LGBMRegressor(
        n_estimators=100,
        num_leaves=15,
        min_child_samples=5,
        colsample_bytree=0.3,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(prepared["X_train_scaled"], prepared["y_train"])
    y_pred = model.predict(prepared["X_test_scaled"])

    return {
        "train_df": prepared["train_df"],
        "test_df": prepared["test_df"],
        "sampled_train_df": prepared["sampled_train_df"],
        "X_train": prepared["X_train"],
        "y_train": prepared["y_train"],
        "X_test": prepared["X_test"],
        "y_test": prepared["y_test"],
        "X_train_scaled": prepared["X_train_scaled"],
        "X_test_scaled": prepared["X_test_scaled"],
        "y_pred": y_pred,
        "spearman": safe_spearman(prepared["y_test"], y_pred),
        "mse": float(np.mean((prepared["y_test"] - y_pred) ** 2)),
    }


def run_outer_fold_tabdpt(
    assay_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int,
    outer_fold: int,
    rng: np.random.Generator,
    random_state: int = 0,
) -> dict[str, object]:
    """Run one TabDPT outer-fold evaluation on a single assay."""
    prepared = prepare_outer_fold_data(
        assay_df,
        feature_cols,
        k=k,
        outer_fold=outer_fold,
        rng=rng,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.random.manual_seed(random_state)
    model = TabDPTRegressor(device=device)
    n_train = len(prepared["X_train_scaled"])
    if n_train < model.max_features:
        raise ValueError("TabDPTRegressor requires n_samples >= max_features")
    model.fit(prepared["X_train_scaled"], prepared["y_train"])
    # pass context_size=n_train so we always use the single-pass path
    y_pred = model.predict(prepared["X_test_scaled"], context_size=n_train)

    return {
        "train_df": prepared["train_df"],
        "test_df": prepared["test_df"],
        "sampled_train_df": prepared["sampled_train_df"],
        "X_train": prepared["X_train"],
        "y_train": prepared["y_train"],
        "X_test": prepared["X_test"],
        "y_test": prepared["y_test"],
        "X_train_scaled": prepared["X_train_scaled"],
        "X_test_scaled": prepared["X_test_scaled"],
        "y_pred": y_pred,
        "spearman": safe_spearman(prepared["y_test"], y_pred),
        "mse": float(np.mean((prepared["y_test"] - y_pred) ** 2)),
    }


def get_feasible_k_values(assay_df: pd.DataFrame, k_sweep: list[int]) -> list[int]:
    """Return feasible k values for this assay, plus -1 for the full-data point.

    A k is feasible if every outer training pool (4/5 of data) has at least k rows.
    """
    min_train_size = min(
        len(assay_df.loc[assay_df[OUTER_FOLD_COL] != f]) for f in OUTER_FOLDS
    )
    return [k for k in k_sweep if k <= min_train_size] + [-1]


def run_assay_benchmark(
    assay_df: pd.DataFrame,
    feature_cols: list[str],
    file_id: str,
    *,
    models: list[str],
    k_sweep: list[int] = K_SWEEP,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run the full k × fold sweep for one assay. Returns a per-row metrics DataFrame."""
    feasible_ks = get_feasible_k_values(assay_df, k_sweep)
    rows = []

    k_label = lambda k: "all" if k == -1 else str(k)
    for k in feasible_ks:
        print(f"  k={k_label(k)}", flush=True)
        for outer_fold in OUTER_FOLDS:
            # Seed per (file_id, k, outer_fold) for reproducibility across partial runs
            rng = np.random.default_rng(
                [random_state, hash(file_id) % (2**31), k % (2**31), outer_fold]
            )

            for model_name in models:
                if (
                    model_name == "tabdpt"
                    and k != -1
                    and k < TABDPT_DEFAULT_MAX_FEATURES
                ):
                    continue
                print(f"    fold={outer_fold} model={model_name}", flush=True)
                try:
                    if model_name == "ridge":
                        result = run_outer_fold_ridge(
                            assay_df,
                            feature_cols,
                            k=k,
                            outer_fold=outer_fold,
                            rng=rng,
                            alpha_grid=ALPHA_GRID,
                            random_state=random_state,
                        )
                    elif model_name == "lgbm":
                        result = run_outer_fold_lightgbm(
                            assay_df,
                            feature_cols,
                            k=k,
                            outer_fold=outer_fold,
                            rng=rng,
                            random_state=random_state,
                        )
                    elif model_name == "tabdpt":
                        result = run_outer_fold_tabdpt(
                            assay_df,
                            feature_cols,
                            k=k,
                            outer_fold=outer_fold,
                            rng=rng,
                            random_state=random_state,
                        )
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                except Exception as e:
                    print(
                        f"  SKIP {file_id} k={k} fold={outer_fold} model={model_name}: {e}"
                    )
                    continue

                rows.append(
                    {
                        "file_id": file_id,
                        "model": model_name,
                        "k": k,
                        "outer_fold": outer_fold,
                        "spearman": result["spearman"],
                        "mse": result["mse"],
                        "n_outer_train": len(result["train_df"]),
                        "n_outer_test": len(result["test_df"]),
                        "n_sampled_train": len(result["sampled_train_df"]),
                        "best_alpha": result.get("best_alpha", float("nan")),
                    }
                )

    return pd.DataFrame(rows)


def run_benchmark(
    ref: pd.DataFrame,
    output_dir: Path,
    *,
    models: list[str] = ("ridge", "lgbm"),
    k_sweep: list[int] = K_SWEEP,
    assay_ids: list[str] | None = None,
    assay_limit: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run the full benchmark across all assays and save metrics to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if assay_ids is not None:
        assay_ids = [a for a in assay_ids if a in ref.index]
    else:
        assay_ids = list(ref.index)
        if assay_limit is not None:
            assay_ids = assay_ids[:assay_limit]

    meta_cols = [
        "UniProt_ID",
        "taxon",
        "seq_len",
        "selection_type",
        "coarse_selection_type",
    ]

    all_rows = []
    for i, file_id in enumerate(assay_ids):
        print(f"[{i + 1}/{len(assay_ids)}] {file_id}")
        try:
            assay_df = load_assay_data(file_id)
        except Exception as e:
            print(f"  SKIP (load failed): {e}")
            continue

        feature_cols = get_feature_columns(assay_df)
        metrics_df = run_assay_benchmark(
            assay_df,
            feature_cols,
            file_id,
            models=list(models),
            k_sweep=k_sweep,
            random_state=random_state,
        )

        # Join assay metadata
        meta_row = ref.loc[file_id]
        for col in meta_cols:
            if col in meta_row.index:
                metrics_df[col] = meta_row[col]

        all_rows.append(metrics_df)

    results = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    models_str = "_".join(sorted(models))
    out_path = (
        output_dir / f"kshot_metrics_{models_str}_{len(assay_ids)}_assays.parquet"
    )
    results.to_parquet(out_path, index=False)
    print(f"\nSaved {len(results)} rows → {out_path}")
    return results


def main() -> None:
    models = ["ridge", "lgbm"]
    output_dir = Path("data/proteingym/results")
    assay_limit = None  # ignored if assay_ids is set
    assay_ids = None  # e.g. ["ASSAY_1", "ASSAY_2"] to target specific assays
    random_state = 0

    ref = load_reference_metadata()
    run_benchmark(
        ref,
        output_dir=output_dir,
        models=models,
        assay_ids=assay_ids,
        assay_limit=assay_limit,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
