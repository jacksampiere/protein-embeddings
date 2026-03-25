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

Generate per-residue embeddings:
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
- `mutant`: single-substitution mutation string, e.g. `A42V`
- `L_res`: length of the WT sequence (number of residues)
- `DMS_score`: experimental fitness score (continuous)
- `DMS_score_bin`: binarized fitness label

Mutation identity + position features:
- `pos_mean_norm`: mutation position normalized by sequence length
- `aa_from_<AA>` (x20): one-hot encoding of the original amino acid at the mutation site
- `aa_to_<AA>` (x20): one-hot encoding of the resulting amino acid at the mutation site
- `delta_hydro`: hydrophobicity delta (mutant - original AA, Kyte-Doolittle)
- `delta_vol`: volume delta (mutant - original AA, Zamyatnin)
- `delta_charge`: formal charge delta (mutant - original AA, at pH 7)

Local WT embeddings:
- `site_mean_<d>`: element `d` of the WT residue embedding at the mutation site
- `win_mean_<d>`: element `d` of the mean-pooled WT embeddings from the local window (length `2k + 1`) around the mutation site
