from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_PATH = Path(
    "data/proteingym/results/kshot_metrics_lgbm_ridge_12_assays.parquet"
)
GROUP_BY: str | None = "coarse_selection_type"  # or "taxon", "selection_type", None
# ─────────────────────────────────────────────────────────────────────────────


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path)
    df["k_label"] = df["k"].apply(lambda x: "all" if x == -1 else str(x))
    # stable x-axis order: fixed k values ascending, then "all"
    k_order = [str(k) for k in sorted(df.loc[df["k"] != -1, "k"].unique())] + ["all"]
    df["k_label"] = pd.Categorical(df["k_label"], categories=k_order, ordered=True)
    # store output dir as df attribute so plot functions can use it
    out_dir = path.parent / path.stem
    out_dir.mkdir(exist_ok=True)
    df.attrs["out_dir"] = out_dir
    return df


def plot_spearman_vs_k(df: pd.DataFrame, group_by: str | None = GROUP_BY) -> None:
    """Plot mean assay-level Spearman vs k, one line per model, optionally faceted by group_by.

    Fixed k values are plotted as connected lines. The 'all' point is drawn as a
    horizontal dashed line per model.
    """
    fixed = df[df["k"] != -1].copy()
    all_data = df[df["k"] == -1].copy()

    group_cols = ["model", "k_label"] + ([group_by] if group_by else [])
    assay_agg = (
        fixed.groupby(
            ["file_id", "model", "k_label"] + ([group_by] if group_by else [])
        )["spearman"]
        .mean()
        .reset_index()
    )
    summary = (
        assay_agg.groupby(group_cols)["spearman"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )

    assay_agg_all = (
        all_data.groupby(["file_id", "model"] + ([group_by] if group_by else []))[
            "spearman"
        ]
        .mean()
        .reset_index()
    )
    summary_all = (
        assay_agg_all.groupby(["model"] + ([group_by] if group_by else []))["spearman"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )

    groups = summary[group_by].unique() if group_by else [None]
    n_groups = len(groups)
    fig, axes = plt.subplots(
        1, n_groups, figsize=(6 * n_groups, 4), sharey=True, squeeze=False
    )

    models = summary["model"].unique()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_color = {m: colors[i] for i, m in enumerate(models)}

    for ax, grp in zip(axes[0], groups):
        sub = summary[summary[group_by] == grp] if group_by else summary
        sub_all = summary_all[summary_all[group_by] == grp] if group_by else summary_all

        for model, mdf in sub.groupby("model"):
            mdf = mdf.sort_values("k_label")
            ax.plot(
                mdf["k_label"],
                mdf["mean"],
                marker="o",
                label=model,
                color=model_color[model],
            )
            ax.fill_between(
                mdf["k_label"],
                mdf["mean"] - mdf["sem"],
                mdf["mean"] + mdf["sem"],
                alpha=0.15,
                color=model_color[model],
            )

        for model, mdf in sub_all.groupby("model"):
            ax.axhline(
                mdf["mean"].values[0],
                linestyle="--",
                linewidth=1.2,
                color=model_color[model],
                alpha=0.7,
                label=f"{model} (all)",
            )

        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlabel("k (training examples)")
        ax.set_ylabel("mean Spearman")
        ax.set_title(str(grp) if grp is not None else "all assays")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("k-shot benchmark: Spearman vs k", fontsize=13)
    fig.tight_layout()
    out_path = df.attrs["out_dir"] / "spearman_vs_k.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def plot_spearman_vs_k_per_assay(df: pd.DataFrame) -> None:
    """One panel per assay, one line per model."""
    assays = sorted(df["file_id"].unique())
    n = len(assays)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False, squeeze=False
    )

    models = df["model"].unique()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_color = {m: colors[i] for i, m in enumerate(models)}

    for ax, assay in zip(axes.flat, assays):
        sub = df[df["file_id"] == assay]
        fold_agg = sub.groupby(["model", "k_label"])["spearman"].mean().reset_index()
        for model, mdf in fold_agg.groupby("model"):
            mdf = mdf.sort_values("k_label")
            ax.plot(
                mdf["k_label"],
                mdf["spearman"],
                marker="o",
                label=model,
                color=model_color[model],
            )
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(assay.split("_")[1], fontsize=9)  # short label
        ax.set_xlabel("k")
        ax.set_ylabel("Spearman")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("k-shot benchmark: per-assay Spearman vs k", fontsize=13)
    fig.tight_layout()
    out_path = df.attrs["out_dir"] / "spearman_vs_k_per_assay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def rank_assays_by_lgbm_ridge_delta(
    df: pd.DataFrame,
    top_n: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Rank assays by mean Spearman delta (lgbm - ridge) across all k and folds.

    Higher delta = lgbm benefits more over ridge = more nonlinear fitness landscape.
    """
    pivot = (
        df.groupby(["file_id", "model"])["spearman"]
        .mean()
        .unstack("model")
        .reset_index()
    )
    pivot["lgbm_minus_ridge"] = pivot["lgbm"] - pivot["ridge"]
    ranked = pivot.sort_values("lgbm_minus_ridge", ascending=False).reset_index(
        drop=True
    )

    if top_n is not None:
        ranked = ranked.head(top_n)

    print(
        ranked[["file_id", "ridge", "lgbm", "lgbm_minus_ridge"]].to_string(index=False)
    )

    if output_path is not None:
        ids = ranked["file_id"].tolist()
        Path(output_path).write_text("\n".join(ids) + "\n")
        print(f"\nWrote {len(ids)} assay IDs → {output_path}")

    return ranked


if __name__ == "__main__":
    df = load_results()
    print(
        f"Loaded {len(df)} rows | assays: {df['file_id'].nunique()} | models: {df['model'].unique()}"
    )

    plot_spearman_vs_k(df)

    # per-assay plot
    plot_spearman_vs_k_per_assay(df)

    # rank assays by lgbm - ridge delta (proxy for nonlinearity)
    # set top_n to limit output, output_path to save IDs for tabdpt targeting
    # rank_assays_by_lgbm_ridge_delta(df, top_n=10, output_path=Path("data/proteingym/results/tabdpt_target_assays.txt"))
