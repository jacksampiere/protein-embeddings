"""Run one ESM-2 forward pass per WT sequence and cache reusable features.

Inputs:
- FASTA file at ``data/wt_fasta`` with one WT sequence per ProteinGym assay/protein ID.

Outputs:
- Per-residue embeddings to ``data/embeddings/wt_tok/{pid}.npy`` with shape ``(L_res, D)``.
- Pooled feature bundles to ``data/proteingym/embeddings/wt_pool/{pid}.npz`` containing
  ``bos``, ``eos``, ``mean``, ``max``, ``seg_mean``, ``len``, and ``pid``.

Notes:
- Residue-level tensors exclude PAD/BOS/EOS positions.
- All pooled features are computed in the same forward pass used for per-residue output.
"""

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FASTA = Path("data/proteingym/wt_fasta")

OUT_TOK = Path("data/proteingym/embeddings/wt_tok")
OUT_TOK.mkdir(parents=True, exist_ok=True)

OUT_POOL = Path("data/proteingym/embeddings/wt_pool")
OUT_POOL.mkdir(parents=True, exist_ok=True)

ESM2_EMBEDDING_IDX = 33  # model has 33 transformer layers
MAX_TOKENS = 6000  # conservative; increase if stable
N_BINS = 8

SAVE_TOK_DTYPE = "float16"
SAVE_POOL_DTYPE = "float16"

TOK_NP_DTYPE = np.dtype(SAVE_TOK_DTYPE)
POOL_NP_DTYPE = np.dtype(SAVE_POOL_DTYPE)


def read_fasta(path: Path):
    """Packs FASTA entries into a list of tuples of file name and sequence."""
    items = []
    cur_id, cur_seq = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                items.append((cur_id, "".join(cur_seq)))
            cur_id = line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_id is not None:
        items.append((cur_id, "".join(cur_seq)))
    return items


def get_batches(items, max_tokens):
    """Pack as many sequences into a batch as possible such that the approximate total token count remains less than `max_tokens`.

    Note that we use a fixed number of bins per protein instead of fixed segment length.
    This is because many functional patterns are relative-position (N-terminus signal peptides,
    C-terminal tails, domain order). Fixed-count bins preserves “where along the protein” consistently.
    """
    i = 0
    while i < len(items):
        batch = []
        j = i
        while j < len(items):
            L = len(items[j][1]) + 2  # ESM-2 contains BOS and EOS tokens
            if (
                batch and (len(batch) + 1) * L > max_tokens
            ):  # approximate padded token count as batch_size * longest_seq_len
                break
            batch.append(items[j])
            j += 1
        yield batch
        i = j


def segment_mean_pool(residue_repr: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Compute contiguous near-equal segment means; empty bins stay zero."""
    if residue_repr.ndim != 2:
        raise ValueError(
            f"Expected (L_res, D) tensor, got shape {tuple(residue_repr.shape)}"
        )

    L_res, D = residue_repr.shape
    seg = torch.zeros((n_bins, D), dtype=residue_repr.dtype, device=residue_repr.device)
    if L_res == 0:
        return seg

    for i in range(n_bins):
        start = (i * L_res) // n_bins
        end = ((i + 1) * L_res) // n_bins
        if end > start:
            seg[i] = residue_repr[start:end].mean(dim=0)
    return seg


data = read_fasta(FASTA)
print(f"Read {len(data)} WT sequences")
data = sorted(data, key=lambda x: len(x[1]))
num_batches = sum(1 for _ in get_batches(data, MAX_TOKENS))
print(f"Planned batches: {num_batches}")

# Load ESM-2
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model.eval()

device = (
    "mps" if torch.backends.mps.is_available() else "cpu"
)  # modify if cuda is an option
model = model.to(device)

# The batch converter is the ESM-2 tokenizer + batch collator
batch_converter = alphabet.get_batch_converter()
pad = alphabet.padding_idx

with torch.no_grad():
    run_start = time.perf_counter()
    proteins_done = 0
    batch_times = []

    pbar = tqdm(
        get_batches(data, MAX_TOKENS),
        total=num_batches,
        desc="Embedding batches",
    )
    for batch_idx, batch in enumerate(pbar, start=1):
        batch_start = time.perf_counter()
        # Each character in the sequence gets an integer token ID
        _, _, toks = batch_converter(batch)  # (B, L)
        toks = toks.to(device)

        # Forward pass maps integer token ID to embeddings
        out = model(toks, repr_layers=[ESM2_EMBEDDING_IDX], return_contacts=False)
        h = out["representations"][ESM2_EMBEDDING_IDX]  # (B, L, D)

        # Valid residues mask: not pad, not BOS, not EOS
        nonpad_mask = toks != pad
        token_lens = nonpad_mask.sum(dim=1)  # includes BOS and EOS
        rows = torch.arange(nonpad_mask.size(0), device=nonpad_mask.device)
        eos_cols = token_lens - 1  # get EOS index for each row
        residue_mask = nonpad_mask.clone()
        residue_mask[:, 0] = False  # mask BOS
        residue_mask[rows, eos_cols] = False  # mask EOS

        # Mean pool over residues (valid positions only)
        mask_f = residue_mask.unsqueeze(-1).type_as(h)
        pooled_mean = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(
            min=1.0
        )  # (B, D)

        # Max pool over residues by masking invalid positions to -inf.
        neg_inf = torch.tensor(float("-inf"), device=h.device, dtype=h.dtype)
        h_for_max = h.masked_fill(~residue_mask.unsqueeze(-1), neg_inf)
        pooled_max = h_for_max.max(dim=1).values
        pooled_max = torch.where(
            torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max)
        )  # edge-case if all invalid

        bos = h[:, 0, :]  # (B, D)
        eos = h[rows, eos_cols, :]  # (B, D)

        # Iterate through each sequence in the current batch and save per-protein artifacts
        for b, (pid, seq) in enumerate(batch):
            residue_idx = torch.where(residue_mask[b])[0]
            residue_repr = h[b, residue_idx, :]  # (L_res, D)
            L_res = residue_repr.size(0)

            if L_res != len(seq):
                raise ValueError(
                    f"Residue alignment mismatch for {pid}: L_res={L_res}, len(seq)={len(seq)}"
                )

            tok_mat = (
                residue_repr.detach().cpu().numpy().astype(TOK_NP_DTYPE, copy=False)
            )
            np.save(OUT_TOK / f"{pid}.npy", tok_mat)
            mean_vec = (
                pooled_mean[b].detach().cpu().numpy().astype(POOL_NP_DTYPE, copy=False)
            )

            max_vec = (
                pooled_max[b].detach().cpu().numpy().astype(POOL_NP_DTYPE, copy=False)
            )
            bos_vec = bos[b].detach().cpu().numpy().astype(POOL_NP_DTYPE, copy=False)
            eos_vec = eos[b].detach().cpu().numpy().astype(POOL_NP_DTYPE, copy=False)
            seg_mean = (
                segment_mean_pool(residue_repr, N_BINS)
                .detach()
                .cpu()
                .numpy()
                .astype(POOL_NP_DTYPE, copy=False)
            )

            np.savez(
                OUT_POOL / f"{pid}.npz",
                bos=bos_vec,
                eos=eos_vec,
                mean=mean_vec,
                max=max_vec,
                seg_mean=seg_mean,
                len=np.asarray(L_res, dtype=np.int32),
                pid=np.asarray(pid, dtype=np.bytes_),
            )
            proteins_done += 1

        batch_sec = time.perf_counter() - batch_start
        batch_times.append(batch_sec)
        avg_batch_sec = sum(batch_times) / len(batch_times)
        eta_sec = avg_batch_sec * (num_batches - batch_idx)
        pbar.set_postfix(
            batch_s=f"{batch_sec:.1f}",
            avg_batch_s=f"{avg_batch_sec:.1f}",
            eta_min=f"{eta_sec / 60.0:.1f}",
            prots_done=proteins_done,
        )

    total_sec = time.perf_counter() - run_start

print(f"Saved per-residue embeddings to {OUT_TOK}")
print(f"Saved per-protein WT pooled features to {OUT_POOL}")
print(f"Total proteins processed: {proteins_done}")
print(f"Total runtime: {total_sec / 60.0:.2f} minutes ({total_sec:.1f} seconds)")
