"""Featurize ProteinGym assays into per-variant tabular rows.

Usage:
    python scripts/featurize.py configs/featurize_baseline.yaml
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

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

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class FeaturizeConfig:
    """Configuration for a featurization run.

    Paths
    -----
    name            : Experiment identifier; output parquets go to ``out_base_dir/name/``.
    ref_csv         : DMS_Substitutions.csv with WT sequences and metadata.
    assay_dir       : Directory of per-assay CSVs (<file_id>.csv) with mutant/score columns.
    wt_tok_dir      : Per-residue WT embeddings (<file_id>.npy), shape (L_res, D).
    wt_special_dir  : BOS/EOS token embeddings (<file_id>.npz), keys ``bos`` and ``eos``.
    out_base_dir    : Root output directory; a subdirectory named ``name`` is created here.

    Data
    ----
    include_multi_mutants : If True, include colon-separated multi-site mutants (e.g. A12V:D45N).
                            Sites are aggregated with mean/max to keep feature width fixed.
    strict                : If True, raise on any filtered row instead of printing a warning.

    Global WT embedding features  (one vector per protein, broadcast to all variant rows)
    -----------------------------
    use_bos       : Include the BOS token embedding as a global protein representation (D dims).
    use_eos       : Include the EOS token embedding as a global protein representation (D dims).
    use_global_mean : Include elementwise mean over all residue embeddings (D dims).
    use_global_max  : Include elementwise max over all residue embeddings (D dims).
    use_global_seg  : Divide the sequence into ``n_seg_bins`` contiguous segments and include
                      the mean embedding of each segment (n_seg_bins × D dims). Captures
                      coarse positional context (N-terminus vs C-terminus, domain ordering).
    n_seg_bins      : Number of equal-length segments for ``use_global_seg``.

    Local WT embedding features  (computed per variant at the mutated site(s))
    ----------------------------
    use_local_site   : site_mean / site_max over the exact mutated residue embedding(s) (D dims each).
    use_local_window : win_mean / win_max over per-site window means R_wt[pos_i±k] (D dims each).
    window_k         : Half-width of the local context window; total window = 2k+1 residues.

    Mutation descriptors  (scalar features derived from the mutation string)
    --------------------
    use_mutation_descriptors : Include n_sites, pos_mean_norm, pos_span_norm, and per-AA
                                counts (aa_from_* and aa_to_*, 20 dims each).
    use_blosum               : Include BLOSUM62(aa_from, aa_to) mean and max over sites.
                                Encodes evolutionary substitutability.
    use_physchem             : Include mean/max delta-hydrophobicity (Kyte-Doolittle),
                                delta-volume (Zamyatnin), and delta-charge over sites.
                                Encodes direct physicochemical perturbation.
    """

    name: str
    ref_csv: str
    assay_dir: str
    wt_tok_dir: str
    wt_special_dir: str
    out_base_dir: str
    include_multi_mutants: bool
    strict: bool
    use_bos: bool
    use_eos: bool
    use_global_mean: bool
    use_global_max: bool
    use_global_seg: bool
    n_seg_bins: int
    use_local_site: bool
    use_local_window: bool
    window_k: int
    use_mutation_descriptors: bool
    use_blosum: bool
    use_physchem: bool


def load_config(path: str) -> FeaturizeConfig:
    with open(path) as f:
        return FeaturizeConfig(**yaml.safe_load(f))


# ── Helpers ───────────────────────────────────────────────────────────────────


def d_cols(prefix: str, d: int) -> list[str]:
    """Configure column names for elements of embedding vectors."""
    return [f"{prefix}_d{i:04d}" for i in range(1, d + 1)]


def seg_cols(n_bins: int, d: int) -> list[str]:
    """Configure column names for elements of segment embedding vectors."""
    return [
        f"wt_seg{b}_d{i:04d}" for b in range(1, n_bins + 1) for i in range(1, d + 1)
    ]


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
        The mutant string could not be parsed — either a token doesn't match
        the pattern ``[AA][pos][AA]`` (e.g. malformed position, missing residue
        letter) or one of the amino acids is not in the canonical 20-AA alphabet.
        Applies to both single mutants and individual tokens within multi-mutants.

    rows_site_invalid
        A site within the mutant passed format parsing but failed a data
        consistency check. Covers three sub-cases (any one fails the whole mutant):
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


def parse_mutant(s: str) -> list[tuple[str, int, str]] | None:
    """Parse 'A123V' or 'A123V:D45N' → [(aa_from, pos, aa_to), ...]. None on failure."""
    sites = []
    for tok in s.split(":"):
        m = MUT_RE.match(tok.strip())
        if not m:
            return None
        af, pos, at = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        if af not in AA_SET or at not in AA_SET:
            return None
        sites.append((af, pos, at))
    return sites or None


def segment_means(tok: np.ndarray, n_bins: int) -> np.ndarray:
    """(L_res, D) float32 → (n_bins, D) float32 contiguous segment mean pool."""
    L, D = tok.shape
    out = np.zeros((n_bins, D), dtype=np.float32)
    for b in range(n_bins):
        s, e = (b * L) // n_bins, ((b + 1) * L) // n_bins
        if e > s:
            out[b] = tok[s:e].mean(axis=0)
    return out


def explode_sites(parsed: pd.Series) -> pd.DataFrame:
    """Expand parsed mutant lists into one row per (mutant_idx, site)."""
    records = [
        (i, af, pos, at) for i, sites in enumerate(parsed) for af, pos, at in sites
    ]
    return pd.DataFrame(records, columns=["mutant_idx", "aa_from", "pos", "aa_to"])


# ── Per-assay processing ──────────────────────────────────────────────────────


def process_one_assay(
    ref_row: pd.Series,
    cfg: FeaturizeConfig,
    blosum,
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

    # ── Global WT features (one vector per protein, broadcast to all rows) ────

    global_cols: list[str] = []
    global_parts: list[np.ndarray] = []

    if cfg.use_bos or cfg.use_eos:
        with np.load(Path(cfg.wt_special_dir) / f"{file_id}.npz") as sp:
            if cfg.use_bos:
                global_cols += d_cols("wt_bos", D)
                global_parts.append(np.asarray(sp["bos"], dtype=np.float16).reshape(-1))
            if cfg.use_eos:
                global_cols += d_cols("wt_eos", D)
                global_parts.append(np.asarray(sp["eos"], dtype=np.float16).reshape(-1))

    if cfg.use_global_mean:
        global_cols += d_cols("wt_mean", D)
        global_parts.append(tok_f32.mean(axis=0).astype(np.float16))

    if cfg.use_global_max:
        global_cols += d_cols("wt_max", D)
        global_parts.append(tok_f32.max(axis=0).astype(np.float16))

    if cfg.use_global_seg:
        global_cols += seg_cols(cfg.n_seg_bins, D)
        global_parts.append(
            segment_means(tok_f32, cfg.n_seg_bins).astype(np.float16).reshape(-1)
        )

    global_row = np.concatenate(global_parts) if global_parts else None

    # ── Load assay CSV and parse mutants ──────────────────────────────────────

    assay_df = pd.read_csv(
        Path(cfg.assay_dir) / f"{file_id}.csv",
        usecols=["mutant", "DMS_score", "DMS_score_bin"],
    )
    assay_df["mutant"] = assay_df["mutant"].astype("string").str.strip()
    stats["rows_seen"] += len(assay_df)

    if not cfg.include_multi_mutants:
        multi_mask = assay_df["mutant"].str.contains(":", regex=False, na=False)
        stats["rows_multi_skipped"] += int(multi_mask.sum())
        assay_df = assay_df.loc[~multi_mask].copy()

    parsed = assay_df["mutant"].map(parse_mutant)
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

    # ── Explode to per-site rows and validate ─────────────────────────────────

    sites_df = explode_sites(parsed)
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
        sites_df = explode_sites(parsed)

    if assay_df.empty:
        stats["assays_skipped_empty"] += 1
        print(f"[SKIP] {file_id}: no valid rows")
        return

    assay_df["DMS_score"] = pd.to_numeric(assay_df["DMS_score"], errors="coerce")
    assay_df["DMS_score_bin"] = pd.to_numeric(
        assay_df["DMS_score_bin"], errors="coerce"
    )
    n_mutants = len(assay_df)

    # ── Per-site embedding lookups ────────────────────────────────────────────

    row_idx = sites_df["mutant_idx"].values  # (n_total_sites,) → which mutant row
    idx0 = sites_df["pos"].values - 1  # 0-indexed residue positions
    site_emb = tok_f32[idx0]  # (n_total_sites, D)

    if cfg.use_local_window:
        prefix = np.vstack(
            [np.zeros((1, D), dtype=np.float32), np.cumsum(tok_f32, axis=0)]
        )
        k = cfg.window_k
        w_start = np.maximum(idx0 - k, 0)
        w_end = np.minimum(idx0 + k + 1, l_res)
        win_emb = (prefix[w_end] - prefix[w_start]) / (w_end - w_start)[:, None]

    # Always needed for descriptor normalization
    site_counts = np.bincount(row_idx, minlength=n_mutants)

    def agg_mean(emb: np.ndarray) -> np.ndarray:
        out = np.zeros((n_mutants, D), dtype=np.float32)
        np.add.at(out, row_idx, emb)
        return out / site_counts[:, None]

    def agg_max(emb: np.ndarray) -> np.ndarray:
        out = np.full((n_mutants, D), -np.inf, dtype=np.float32)
        np.maximum.at(out, row_idx, emb)
        return out

    # ── Assemble output DataFrame ─────────────────────────────────────────────

    out_dfs: list[pd.DataFrame] = []

    # Meta + labels
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

    # Mutation descriptors
    need_aa_lists = cfg.use_mutation_descriptors or cfg.use_blosum or cfg.use_physchem
    if need_aa_lists:
        from_aas = sites_df["aa_from"].tolist()
        to_aas = sites_df["aa_to"].tolist()

    if cfg.use_mutation_descriptors:
        pos_arr = sites_df["pos"].values.astype(np.float32)
        pos_sum = np.zeros(n_mutants, dtype=np.float32)
        pos_min = np.full(n_mutants, np.inf, dtype=np.float32)
        pos_max_agg = np.full(n_mutants, -np.inf, dtype=np.float32)
        np.add.at(pos_sum, row_idx, pos_arr)
        np.minimum.at(pos_min, row_idx, pos_arr)
        np.maximum.at(pos_max_agg, row_idx, pos_arr)
        pos_span = pos_max_agg - pos_min
        pos_span[site_counts == 1] = 0.0

        from_idx = np.array([AA_INDEX[a] for a in from_aas], dtype=np.int32)
        to_idx = np.array([AA_INDEX[a] for a in to_aas], dtype=np.int32)
        aa_from_counts = np.zeros((n_mutants, 20), dtype=np.uint8)
        aa_to_counts = np.zeros((n_mutants, 20), dtype=np.uint8)
        np.add.at(aa_from_counts, (row_idx, from_idx), 1)
        np.add.at(aa_to_counts, (row_idx, to_idx), 1)

        out_dfs.append(
            pd.DataFrame(
                {
                    "n_sites": site_counts.astype(np.int16),
                    "pos_mean_norm": (pos_sum / site_counts / l_res).astype(np.float32),
                    "pos_span_norm": (pos_span / l_res).astype(np.float32),
                }
            )
        )
        out_dfs.append(
            pd.DataFrame(aa_from_counts, columns=[f"aa_from_{aa}" for aa in AA_ORDER])
        )
        out_dfs.append(
            pd.DataFrame(aa_to_counts, columns=[f"aa_to_{aa}" for aa in AA_ORDER])
        )

    if cfg.use_blosum:
        scores = np.array(
            [blosum[af, at] for af, at in zip(from_aas, to_aas)], dtype=np.float32
        )
        b_sum = np.zeros(n_mutants, dtype=np.float32)
        b_max = np.full(n_mutants, -np.inf, dtype=np.float32)
        np.add.at(b_sum, row_idx, scores)
        np.maximum.at(b_max, row_idx, scores)
        out_dfs.append(
            pd.DataFrame(
                {
                    "blosum_mean": b_sum / site_counts,
                    "blosum_max": b_max,
                }
            )
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
            d_sum = np.zeros(n_mutants, dtype=np.float32)
            d_max = np.full(n_mutants, -np.inf, dtype=np.float32)
            np.add.at(d_sum, row_idx, delta)
            np.maximum.at(d_max, row_idx, delta)
            out_dfs.append(
                pd.DataFrame(
                    {
                        f"{col_prefix}_mean": d_sum / site_counts,
                        f"{col_prefix}_max": d_max,
                    }
                )
            )

    # Global WT embedding (broadcast to all rows)
    if global_row is not None:
        out_dfs.append(
            pd.DataFrame(
                np.broadcast_to(
                    global_row.reshape(1, -1), (n_mutants, global_row.size)
                ),
                columns=global_cols,
            )
        )

    # Local embedding features
    if cfg.use_local_site:
        out_dfs.append(
            pd.DataFrame(
                agg_mean(site_emb).astype(np.float16), columns=d_cols("site_mean", D)
            )
        )
        out_dfs.append(
            pd.DataFrame(
                agg_max(site_emb).astype(np.float16), columns=d_cols("site_max", D)
            )
        )

    if cfg.use_local_window:
        out_dfs.append(
            pd.DataFrame(
                agg_mean(win_emb).astype(np.float16), columns=d_cols("win_mean", D)
            )
        )
        out_dfs.append(
            pd.DataFrame(
                agg_max(win_emb).astype(np.float16), columns=d_cols("win_max", D)
            )
        )

    out_path = out_dir / f"{file_id}.parquet"
    pd.concat(out_dfs, axis=1, copy=False).to_parquet(
        out_path, index=False, engine="pyarrow"
    )

    stats["assays_written"] += 1
    stats["rows_written"] += n_mutants
    print(f"[OK] {file_id}: {n_mutants} rows → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if len(args) != 1:
        print(f"Usage: python {sys.argv[0]} <config.yaml> [--one]")
        sys.exit(1)

    one = "--one" in flags
    cfg = load_config(args[0])
    out_dir = Path(cfg.out_base_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    blosum = None
    if cfg.use_blosum:
        from Bio.Align import substitution_matrices

        blosum = substitution_matrices.load("BLOSUM62")

    ref = pd.read_csv(cfg.ref_csv)
    ref["file_id"] = ref["DMS_filename"].map(lambda x: Path(x).stem)
    ref_ids = set(ref["file_id"])

    problems = []
    for label, glob_pat, dir_path in [
        ("wt_tok", "*.npy", cfg.wt_tok_dir),
        ("wt_special", "*.npz", cfg.wt_special_dir),
        ("assay", "*.csv", cfg.assay_dir),
    ]:
        missing = sorted(ref_ids - {p.stem for p in Path(dir_path).glob(glob_pat)})
        if missing:
            problems.append(f"Missing {label} files ({len(missing)}): {missing[:10]}")
    if problems:
        raise RuntimeError("\n".join(problems))

    if one:
        ref = ref.iloc[:1]

    print(f"Config '{cfg.name}' | {len(ref)} assays | out={out_dir}")

    stats: defaultdict = defaultdict(int)
    stats["assays_total"] = len(ref)

    for _, ref_row in tqdm(ref.iterrows(), total=len(ref), desc="Featurizing assays"):
        process_one_assay(ref_row, cfg, blosum, stats, out_dir)

    print("\n=== Summary ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
