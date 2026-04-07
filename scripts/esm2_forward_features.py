import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FASTA = Path("data/proteingym/wt_fasta")

OUT_TOK = Path("data/proteingym/embeddings/wt_tok")
OUT_TOK.mkdir(parents=True, exist_ok=True)

ESM2_LAYER = 33  # model has 33 transformer layers
MAX_TOKENS = 6000  # conservative; increase if stable
DTYPE = np.float16


def read_fasta(path: Path):
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
    """Pack sequences into batches up to approximate token budget."""
    i = 0
    while i < len(items):
        batch = []
        j = i
        while j < len(items):
            L = len(items[j][1]) + 2  # +2 for BOS and EOS
            if batch and (len(batch) + 1) * L > max_tokens:
                break
            batch.append(items[j])
            j += 1
        yield batch
        i = j


data = read_fasta(FASTA)
print(f"Read {len(data)} WT sequences")
data = sorted(data, key=lambda x: len(x[1]))
num_batches = sum(1 for _ in get_batches(data, MAX_TOKENS))
print(f"Planned batches: {num_batches}")

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

batch_converter = alphabet.get_batch_converter()
pad = alphabet.padding_idx

with torch.no_grad():
    run_start = time.perf_counter()
    proteins_done = 0
    batch_times = []

    pbar = tqdm(
        get_batches(data, MAX_TOKENS), total=num_batches, desc="Embedding batches"
    )
    for batch_idx, batch in enumerate(pbar, start=1):
        batch_start = time.perf_counter()

        _, _, toks = batch_converter(batch)  # (B, L)
        toks = toks.to(device)

        out = model(toks, repr_layers=[ESM2_LAYER], return_contacts=False)
        h = out["representations"][ESM2_LAYER]  # (B, L, D)

        nonpad_mask = toks != pad
        token_lens = nonpad_mask.sum(dim=1)  # includes BOS and EOS
        rows = torch.arange(h.size(0), device=device)
        eos_cols = token_lens - 1

        residue_mask = nonpad_mask.clone()
        residue_mask[:, 0] = False  # mask BOS
        residue_mask[rows, eos_cols] = False  # mask EOS

        for b, (pid, seq) in enumerate(batch):
            residue_idx = torch.where(residue_mask[b])[0]
            residue_repr = h[b, residue_idx, :]  # (L_res, D)

            if residue_repr.size(0) != len(seq):
                raise ValueError(
                    f"Residue alignment mismatch for {pid}: "
                    f"got {residue_repr.size(0)}, expected {len(seq)}"
                )

            np.save(
                OUT_TOK / f"{pid}.npy",
                residue_repr.cpu().numpy().astype(DTYPE, copy=False),
            )
            proteins_done += 1

        batch_sec = time.perf_counter() - batch_start
        batch_times.append(batch_sec)
        avg = sum(batch_times) / len(batch_times)
        eta = avg * (num_batches - batch_idx)
        pbar.set_postfix(
            batch_s=f"{batch_sec:.1f}",
            avg_s=f"{avg:.1f}",
            eta_min=f"{eta / 60:.1f}",
            done=proteins_done,
        )

    total_sec = time.perf_counter() - run_start

print(f"Saved per-residue embeddings → {OUT_TOK}")
print(f"Proteins processed: {proteins_done}  |  Total time: {total_sec / 60:.2f} min")
