from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Configs
BASE_RESULTS_PATH = Path(
    "data/proteingym/results/kshot_metrics_lgbm_ridge_217_assays.parquet"
)
TABDPT_RESULTS_PATH = Path(
    "data/proteingym/results/kshot_metrics_tabdpt_10_assays.parquet"
)


def load_combined(
    base_path: Path = BASE_RESULTS_PATH,
    tabdpt_path: Path = TABDPT_RESULTS_PATH,
) -> pd.DataFrame:
    """Load ridge/lgbm and tabdpt results, join, and filter to the tabdpt assay subset."""
    base = pd.read_parquet(base_path)
    tabdpt = pd.read_parquet(tabdpt_path)

    tabdpt_assays = tabdpt["file_id"].unique()
    base_subset = base[base["file_id"].isin(tabdpt_assays)]

    df = pd.concat([base_subset, tabdpt], ignore_index=True)

    df["k_label"] = df["k"].apply(lambda x: "all" if x == -1 else str(x))
    k_order = [str(k) for k in sorted(df.loc[df["k"] != -1, "k"].unique())] + ["all"]
    df["k_label"] = pd.Categorical(df["k_label"], categories=k_order, ordered=True)

    out_dir = Path("plots") / Path(tabdpt_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    df.attrs["out_dir"] = out_dir

    return df


def plot_tabdpt_comparison(df: pd.DataFrame) -> None:
    """Aggregate Spearman vs k for all three models on the tabdpt assay subset."""
    fixed = df[df["k"] != -1].copy()
    all_data = df[df["k"] == -1].copy()

    assay_agg = (
        fixed.groupby(["file_id", "model", "k_label"])["spearman"].mean().reset_index()
    )
    summary = (
        assay_agg.groupby(["model", "k_label"])["spearman"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )
    summary_all = (
        all_data.groupby(["file_id", "model"])["spearman"]
        .mean()
        .reset_index()
        .groupby("model")["spearman"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )

    models = summary["model"].unique()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_color = {m: colors[i] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, mdf in summary.groupby("model"):
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

    for model, mdf in summary_all.groupby("model"):
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
    ax.set_title(f"TabDPT vs Ridge vs LightGBM — {df['file_id'].nunique()} assays")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    out_path = df.attrs["out_dir"] / "tabdpt_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def plot_tabdpt_comparison_per_assay(df: pd.DataFrame) -> None:
    """One panel per assay comparing all three models."""
    assays = sorted(df["file_id"].unique())
    n = len(assays)
    # Hardcode 3x3 since we have 9 assays for comparison
    ncols = 3
    nrows = 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False, squeeze=False
    )

    models = df["model"].unique()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_color = {m: colors[i] for i, m in enumerate(models)}

    for ax, assay in zip(axes.flat, assays):
        sub = df[df["file_id"] == assay]
        fixed = sub[sub["k"] != -1]
        all_data = sub[sub["k"] == -1]

        fold_agg = fixed.groupby(["model", "k_label"])["spearman"].mean().reset_index()
        for model, mdf in fold_agg.groupby("model"):
            mdf = mdf.sort_values("k_label")
            ax.plot(
                mdf["k_label"],
                mdf["spearman"],
                marker="o",
                label=model,
                color=model_color[model],
            )

        for model, mdf in all_data.groupby("model"):
            val = mdf["spearman"].mean()
            ax.axhline(
                val, linestyle="--", linewidth=1.0, color=model_color[model], alpha=0.7
            )

        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_title(assay.split("_")[1], fontsize=9)
        ax.set_xlabel("k")
        ax.set_ylabel("Spearman")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("TabDPT vs Ridge vs LightGBM — per assay", fontsize=13)
    fig.tight_layout()
    out_path = df.attrs["out_dir"] / "tabdpt_comparison_per_assay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def print_tabdpt_summary_stats(df: pd.DataFrame) -> None:
    """Print TabDPT vs ridge vs lgbm stats on the targeted assay subset."""
    by_k = (
        df.groupby(["file_id", "model", "k"])["spearman"]
        .mean()
        .reset_index()
        .groupby(["model", "k"])["spearman"]
        .mean()
        .unstack("model")
        .round(3)
    )
    print("=== Mean Spearman by model x k (TabDPT subset) ===")
    print(by_k.to_string())
    print()


if __name__ == "__main__":
    df = load_combined()
    print(
        f"Loaded {len(df)} rows | assays: {df['file_id'].nunique()} | models: {df['model'].unique()}"
    )

    print_tabdpt_summary_stats(df)
    plot_tabdpt_comparison(df)
    plot_tabdpt_comparison_per_assay(df)
