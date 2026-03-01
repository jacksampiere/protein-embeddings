# In-context learning via tabular foundation models with ProteinGym
TODO

## Running the code

### Environment setup

Install uv:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and install the environment:

```shell
uv venv .venv
uv sync
source .venv/bin/activate
```

Check dependencies:
```shell
python -m ensurepip --upgrade  # install pip
python -m pip install --upgrade pip  # upgrade if an old version is installed by default
python -m pip list
```

### Dataset download

Create the dataset directory:

```shell
mkdir -p data/proteingym/reference_files  # from the repo root
```

Download the reference file with the non-mutated sequences by copying the file `DMS_Substitutions.csv` from [here](https://github.com/OATML-Markslab/ProteinGym/tree/37ea726885452197125f841a33320341d665bc3f/reference_files) and pasting it into `data/proteingym/reference_files`.

Download the substitutions and predefined folds:

```shell
VERSION="v1.3"
BASE="https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}"

# Main DMS substitutions benchmark (variants + labels)
curl -L -o DMS_ProteinGym_substitutions.zip "${BASE}/DMS_ProteinGym_substitutions.zip"
unzip -q DMS_ProteinGym_substitutions.zip -d data/proteingym
rm DMS_ProteinGym_substitutions.zip

# Official CV folds for singles
curl -L -o cv_folds_singles_substitutions.zip "${BASE}/cv_folds_singles_substitutions.zip"
unzip -q data/proteingym/cv_folds_singles_substitutions.zip -d data/proteingym
rm cv_folds_singles_substitutions.zip

# Official CV folds for multiple
curl -L -o cv_folds_multiples_substitutions.zip "${BASE}/cv_folds_multiples_substitutions.zip"
unzip -q cv_folds_multiples_substitutions.zip -d data/proteingym
rm cv_folds_multiples_substitutions.zip
```

### Core functionality

Convert WT sequences to FASTA:
```shell
python scripts/extract_wt_fasta.py
```

Generate per-residue embeddings + BOS/EOS tokens:
```shell
python scripts/esm2_forward_features.py
```

Generate features:
```shell
python scripts/featurize.py configs/featurize_baseline.yaml
```

## Feature descriptions:
Metadata:
- `file_id`: assay identifier (stem of DMS_filename, e.g. `A0A140D2T1_ZIKV_Sourisseau_2019`)
- `UniProt_ID`: UniProt accession of the WT protein
- `mutant`: mutation string, e.g. `A42V` or `A42V:L100F` for multi-mutants
- `L_res`: length of the WT sequence (number of residues)
- `DMS_score`: experimental fitness score (continuous)
- `DMS_score_bin`: binarized fitness label

Mutation descriptors:
- `n_sites`: number of mutation sites
- `pos_mean_norm`: mean position of the mutation sites, normalized by sequence length
- `pos_span_norm`: distance from last to first mutation site, normalized by sequence length; zero for single-mutation variants
- `aa_from_<AA>` (x20): number of mutations for which `<AA>` is the original amino acid at the mutation site
- `aa_to_<AA>` (x20): number of mutations for which `<AA>` is the resulting amino acid at the mutation site
- `blosum_mean`: mean BLOSUM62 score across all mutations in the variant
- `blosum_max`: max BLOSUM62 score across all mutations in the variant
- `delta_hydro_mean`: mean hydrophobicity delta (mutant - original AA, Kyte-Doolittle) across all mutation sites
- `delta_hydro_max`: max hydrophobicity delta (mutant - original) across all mutation sites
- `delta_vol_mean`: mean volume delta (mutant - original, Zamyatnin) across all mutation sites
- `delta_vol_max`: max volume delta (mutant - original) across all mutation sites
- `delta_charge_mean`: mean formal charge delta (mutant - original, at pH 7) across all mutation sites
- `delta_charge_max`: max formal charge delta (mutant - original) across all mutation sites

Global WT embeddings:
- `wt_bos_<d>`: element `d` of the original WT sequence BOS token
- `wt_eos_<d>`: element `d` of the original WT sequence EOS token
- `wt_mean_<d>`: element `d` of the original WT sequence embeddings after mean pooling
- `wt_max_<d>`: element `d` of the original WT sequence embeddings after max pooling
- `wt_seg<bin>_<d>`: element `d` of segment `bin` of the original WT sequence embeddings after mean pooling

Local WT embeddings:
- `site_mean_<d>`: element `d` of the WT residue embedding at the mutation site; for multi-mutants, mean-pooled across sites
- `site_max_<d>`: element `d` of the WT residue embedding at the mutation site; for multi-mutants, max-pooled across sites
- `win_mean_<d>`: element `d` of the mean-pooled embeddings from the window (length 2k + 1) around the mutation site(s)
- `win_max_<d>`: element `d` of the max-pooled embeddings from the window (length 2k + 1) around the mutation site(s)
