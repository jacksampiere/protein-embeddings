from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

FASTA = Path("data/wt_fasta")
OUT_MEAN = Path("embeddings/wt_mean")
OUT_MEAN.mkdir(parents=True, exist_ok=True)
OUT_TOK = Path("embeddings/wt_tok")
OUT_TOK.mkdir(parents=True, exist_ok=True)


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


data = read_fasta(FASTA)

print("WT sequences:", len(data))
data = sorted(data, key=lambda x: len(x[1]))

# Load ESM-2
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
pad = alphabet.padding_idx

MAX_TOKENS = 4000  # conservative; increase if stable


def get_batches(items, max_tokens):
    """Pack as many sequences into a batch as possible such that the approximate total token count remains less than `max_tokens`."""
    i = 0
    while i < len(items):
        batch = []
        j = i
        while j < len(items):
            L = len(items[j][1]) + 2  # ESM-2 contains BOS/CLS and EOS tokens
            if (
                batch and (len(batch) + 1) * L > max_tokens
            ):  # approximate padded token count as batch_size * longest_seq_len
                break
            batch.append(items[j])
            j += 1
        yield batch
        i = j


with torch.no_grad():
    for batch in tqdm(list(get_batches(data, MAX_TOKENS)), desc="Embedding batches"):
        labels, strs, toks = batch_converter(batch)  # toks: (B,L)
        toks = toks.to(device)
        out = model(toks, repr_layers=[33], return_contacts=False)
        h = out["representations"][33]  # (B,L,D)

        # valid residues mask: not PAD, not BOS, not EOS
        mask = toks != pad
        mask[:, 0] = False
        for r in range(mask.size(0)):
            idx = torch.where(mask[r])[0]
            if len(idx) > 0:
                mask[r, idx[-1]] = False

        # pooled mean over residues
        mask_f = mask.unsqueeze(-1).type_as(h)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (B,D)

        for b, (pid, seq) in enumerate(batch):
            mean_vec = pooled[b].detach().cpu().numpy().astype(np.float32)
            np.save(OUT_MEAN / f"{pid}.npy", mean_vec)

            idx = torch.where(mask[b])[0]
            tok_mat = (
                h[b, idx, :].detach().cpu().numpy().astype(np.float32)
            )  # (L_res,D)
            np.save(OUT_TOK / f"{pid}.npy", tok_mat)

print("Saved pooled to", OUT_MEAN)
print("Saved per-residue to", OUT_TOK)
