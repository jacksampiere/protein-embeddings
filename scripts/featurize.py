import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Constants
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_ARRAY = np.array(list(AA_ORDER), dtype="<U1")
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_SET = set(AA_ORDER)

MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
MAX_FILTER_EXAMPLES = 5

# Kyte-Doolittle (1982) hydrophobicity
HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# Residue volumes Å³ (Zamyatnin 1972)
VOLUME = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}

# Formal charge at pH 7
CHARGE = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 0,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0,
}


@dataclass
class FeaturizeConfig:
    """Configuration for a featurization run.

    Paths
    -----
    name            : Experiment identifier; output parquets go to ``out_base_dir/name/``.
    ref_csv         : DMS_Substitutions.csv with WT sequences and metadata.
    assay_dir       : Directory of per-assay CSVs (<file_id>.csv) with mutant/score columns.
    wt_tok_dir      : Per-residue WT embeddings (<file_id>.npy), shape (L_res, D).
    out_base_dir    : Root output directory; a subdirectory named ``name`` is created here.

    Data
    ----
    strict : If True, raise on any filtered row instead of printing a warning.

    Local WT embedding features  (computed per variant at the mutated site)
    ----------------------------
    use_local_window : Include win_mean over the WT local window R_wt[pos_i±k] (D dims).
    window_k         : Half-width of the local context window; total window = 2k+1 residues.

    Mutation identity + position features
    -------------------------------------
    use_mutation_identity : Include pos_mean_norm and aa_from/aa_to one-hots.
    use_physchem          : Include delta-hydrophobicity (Kyte-Doolittle),
                            delta-volume (Zamyatnin), and delta-charge.
    """

    name: str
    ref_csv: str
    assay_dir: str
    wt_tok_dir: str
    out_base_dir: str
    strict: bool
    use_local_window: bool
    window_k: int
    use_mutation_identity: bool
    use_physchem: bool


def load_config(path: str) -> FeaturizeConfig:
    with open(path) as f:
        return FeaturizeConfig(**yaml.safe_load(f))


# ------ Helpers ------


def d_cols(prefix: str, d: int) -> list[str]:
    """Configure column names for elements of embedding vectors."""
    return [f"{prefix}_d{i:04d}" for i in range(1, d + 1)]


def apply_filter(
    df: pd.DataFrame,
    keep: pd.Series,
    *,
    file_id: str,
    reason: str,
    strict: bool,
    stats: defaultdict,
) -> pd.DataFrame:
    """Drop rows where ``keep`` is False, log a warning, and update stats.

    Possible reasons a row is dropped
    ----------------------------------
    rows_bad_mutant_format
        The mutant string could not be parsed — it doesn't match the pattern
        ``[AA][pos][AA]`` (e.g. malformed position, missing residue letter)
        or one of the amino acids is not in the canonical 20-AA alphabet.

    rows_site_invalid
        The mutant passed format parsing but failed a data consistency check.
        Covers three sub-cases:
        (1) Position is out of the range [1, L_res] of the WT token array.
        (2) Position exceeds the length of the reference target_seq string.
        (3) The aa_from letter in the mutant string does not match the actual
            residue in target_seq at that position — indicating a mismatch
            between the assay CSV and the reference sequence.
    """
    dropped = int((~keep).sum())
    if dropped == 0:
        return df
    stats[reason] += dropped
    examples = df.loc[~keep, "mutant"].astype(str).head(MAX_FILTER_EXAMPLES).tolist()
    msg = f"{file_id}: dropped {dropped} ({reason}); examples={examples}"
    if strict:
        raise ValueError(msg)
    print(f"[WARN] {msg}")
    return df.loc[keep].copy()


def parse_single_mutant(s: str) -> tuple[str, int, str] | None:
    """Parse 'A123V' → (aa_from, pos, aa_to). Return None on failure."""
    m = MUT_RE.match(s.strip())
    if not m:
        return None
    af, pos, at = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
    if af not in AA_SET or at not in AA_SET:
        return None
    return af, pos, at


# ------ Per-assay processing ------


def process_one_assay(
    ref_row: pd.Series,
    cfg: FeaturizeConfig,
    stats: defaultdict,
    out_dir: Path,
) -> None:
    file_id = str(ref_row["file_id"])
    uniprot_id = str(ref_row.get("UniProt_ID", "")).strip()
    target_seq = str(ref_row.get("target_seq", "")).strip().upper()

    tok = np.load(
        Path(cfg.wt_tok_dir) / f"{file_id}.npy", mmap_mode="r"
    )  # (L_res, D); f16
    l_res, D = tok.shape
    tok_f32 = np.asarray(tok, dtype=np.float32)

    # --- Load assay CSV and keep only single substitutions ---

    assay_df = pd.read_csv(
        Path(cfg.assay_dir) / f"{file_id}.csv",
        usecols=["mutant", "DMS_score", "DMS_score_bin"],
    )
    assay_df["mutant"] = assay_df["mutant"].astype("string").str.strip()
    stats["rows_seen"] += len(assay_df)

    multi_mask = assay_df["mutant"].str.contains(":", regex=False, na=False)
    stats["rows_multi_skipped"] += int(multi_mask.sum())
    assay_df = assay_df.loc[~multi_mask].copy()

    parsed = assay_df["mutant"].map(parse_single_mutant)
    keep = pd.Series(parsed.notna().values, index=assay_df.index)
    assay_df = apply_filter(
        assay_df,
        keep,
        file_id=file_id,
        reason="rows_bad_mutant_format",
        strict=cfg.strict,
        stats=stats,
    )
    parsed = parsed.loc[assay_df.index]
    assay_df = assay_df.reset_index(drop=True)
    parsed = parsed.reset_index(drop=True)

    # --- Parse the single-mutant rows and validate site identity ---

    sites_df = pd.DataFrame(
        parsed.tolist(), columns=["aa_from", "pos", "aa_to"]
    ).reset_index(names="mutant_idx")
    pos_vals = sites_df["pos"].values

    valid = (pos_vals >= 1) & (pos_vals <= l_res)
    if target_seq:
        in_seq = pos_vals <= len(target_seq)
        valid &= in_seq
        ref_aas = np.array(
            [target_seq[p - 1] if ib else "?" for p, ib in zip(pos_vals, in_seq)],
            dtype="<U1",
        )
        valid &= ref_aas == sites_df["aa_from"].to_numpy(dtype="<U1")

    if not valid.all():
        bad = np.zeros(len(assay_df), dtype=bool)
        bad[sites_df.loc[~valid, "mutant_idx"].unique()] = True
        keep = pd.Series(~bad, index=assay_df.index)
        assay_df = apply_filter(
            assay_df,
            keep,
            file_id=file_id,
            reason="rows_site_invalid",
            strict=cfg.strict,
            stats=stats,
        )
        parsed = parsed.loc[assay_df.index]
        assay_df = assay_df.reset_index(drop=True)
        parsed = parsed.reset_index(drop=True)
        sites_df = pd.DataFrame(
            parsed.tolist(), columns=["aa_from", "pos", "aa_to"]
        ).reset_index(names="mutant_idx")

    if assay_df.empty:
        stats["assays_skipped_empty"] += 1
        print(f"[SKIP] {file_id}: no valid rows")
        return

    assay_df["DMS_score"] = pd.to_numeric(assay_df["DMS_score"], errors="coerce")
    assay_df["DMS_score_bin"] = pd.to_numeric(
        assay_df["DMS_score_bin"], errors="coerce"
    )
    n_mutants = len(assay_df)

    # --- Per-site embedding lookups ---

    row_idx = sites_df["mutant_idx"].values  # (n_mutants,) → which mutant row
    idx0 = sites_df["pos"].values - 1  # 0-indexed residue positions
    site_emb = tok_f32[idx0]  # (n_mutants, D)

    if cfg.use_local_window:
        prefix = np.vstack(
            [np.zeros((1, D), dtype=np.float32), np.cumsum(tok_f32, axis=0)]
        )
        k = cfg.window_k
        w_start = np.maximum(idx0 - k, 0)
        w_end = np.minimum(idx0 + k + 1, l_res)
        win_emb = (prefix[w_end] - prefix[w_start]) / (w_end - w_start)[:, None]

    if len(row_idx) != n_mutants:
        raise ValueError(
            f"{file_id}: singles-only featurization expected one site per row"
        )

    # --- Assemble output DataFrame ---

    out_dfs: list[pd.DataFrame] = []

    # Metadata + labels
    out_dfs.append(
        pd.DataFrame(
            {
                "file_id": np.repeat(file_id, n_mutants),
                "UniProt_ID": np.repeat(uniprot_id, n_mutants),
                "mutant": assay_df["mutant"].to_numpy(),
                "L_res": np.full(n_mutants, l_res, dtype=np.int32),
                "DMS_score": assay_df["DMS_score"].to_numpy(dtype=np.float32),
                "DMS_score_bin": assay_df["DMS_score_bin"].to_numpy(dtype=np.float32),
            }
        )
    )

    # Mutation identity + position features
    need_aa_lists = cfg.use_mutation_identity or cfg.use_physchem
    if need_aa_lists:
        from_aas = sites_df["aa_from"].tolist()
        to_aas = sites_df["aa_to"].tolist()

    if cfg.use_mutation_identity:
        pos_arr = sites_df["pos"].values.astype(np.float32)
        from_idx = np.array([AA_INDEX[a] for a in from_aas], dtype=np.int32)
        to_idx = np.array([AA_INDEX[a] for a in to_aas], dtype=np.int32)
        aa_from_counts = np.zeros((n_mutants, 20), dtype=np.uint8)
        aa_to_counts = np.zeros((n_mutants, 20), dtype=np.uint8)
        np.add.at(aa_from_counts, (row_idx, from_idx), 1)
        np.add.at(aa_to_counts, (row_idx, to_idx), 1)

        out_dfs.append(
            pd.DataFrame(
                {
                    "pos_mean_norm": (pos_arr / l_res).astype(np.float32),
                }
            )
        )
        out_dfs.append(
            pd.DataFrame(aa_from_counts, columns=[f"aa_from_{aa}" for aa in AA_ORDER])
        )
        out_dfs.append(
            pd.DataFrame(aa_to_counts, columns=[f"aa_to_{aa}" for aa in AA_ORDER])
        )

    if cfg.use_physchem:
        for col_prefix, table in [
            ("delta_hydro", HYDROPHOBICITY),
            ("delta_vol", VOLUME),
            ("delta_charge", CHARGE),
        ]:
            delta = np.array(
                [table[at] - table[af] for af, at in zip(from_aas, to_aas)],
                dtype=np.float32,
            )
            out_dfs.append(
                pd.DataFrame(
                    {
                        col_prefix: delta,
                    }
                )
            )

    # Local WT embedding at the mutated site
    out_dfs.append(
        pd.DataFrame(site_emb.astype(np.float16), columns=d_cols("site_mean", D))
    )

    if cfg.use_local_window:
        out_dfs.append(
            pd.DataFrame(win_emb.astype(np.float16), columns=d_cols("win_mean", D))
        )

    out_path = out_dir / f"{file_id}.parquet"
    pd.concat(out_dfs, axis=1, copy=False).to_parquet(
        out_path, index=False, engine="pyarrow"
    )

    stats["assays_written"] += 1
    stats["rows_written"] += n_mutants
    print(f"[OK] {file_id}: {n_mutants} rows → {out_path}")


def main() -> None:
    args = sys.argv[1:]

    if len(args) not in {1, 2}:
        print(f"Usage: python {sys.argv[0]} <config.yaml> [n_assays]")
        sys.exit(1)

    cfg = load_config(args[0])
    n_assays = int(args[1]) if len(args) == 2 else None
    out_dir = Path(cfg.out_base_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(cfg.ref_csv)
    ref["file_id"] = ref["DMS_filename"].map(lambda x: Path(x).stem)
    ref_ids = set(ref["file_id"])

    problems = []
    for label, glob_pat, dir_path in [
        ("wt_tok", "*.npy", cfg.wt_tok_dir),
        ("assay", "*.csv", cfg.assay_dir),
    ]:
        missing = sorted(ref_ids - {p.stem for p in Path(dir_path).glob(glob_pat)})
        if missing:
            problems.append(f"Missing {label} files ({len(missing)}): {missing[:10]}")
    if problems:
        raise RuntimeError("\n".join(problems))

    if n_assays is not None:
        ref = ref.iloc[:n_assays]

    print(f"Config '{cfg.name}' | {len(ref)} assays | out={out_dir}")

    stats: defaultdict = defaultdict(int)
    stats["assays_total"] = len(ref)

    for _, ref_row in tqdm(ref.iterrows(), total=len(ref), desc="Featurizing assays"):
        process_one_assay(ref_row, cfg, stats, out_dir)

    print("\n=== Summary ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
