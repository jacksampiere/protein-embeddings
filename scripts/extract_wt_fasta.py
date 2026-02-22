from pathlib import Path
import pandas as pd

ref_path = Path("data/proteingym/reference_files/DMS_Substitutions.csv")
out_fasta = Path("data/proteingym/wt_fasta")
out_fasta.parent.mkdir(parents=True, exist_ok=True)

ref = pd.read_csv(ref_path)

n = 0
with out_fasta.open("w") as f:
    for _, r in ref.iterrows():
        # DMS_filename looks like "A0A1I9GEU1_NEIME_Kennouche_2019.csv"
        pid = Path(r["DMS_filename"]).stem
        wt = str(r["target_seq"]).strip()
        if not wt or wt.lower() == "nan":
            continue
        f.write(f">{pid}\n{wt}\n")
        n += 1

print(f"Wrote {n} WT sequences to {out_fasta}")
