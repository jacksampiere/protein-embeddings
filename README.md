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

 cd data/proteingym

# Main DMS substitutions benchmark (variants + labels)
curl -L -o DMS_ProteinGym_substitutions.zip "${BASE}/DMS_ProteinGym_substitutions.zip"
unzip -q DMS_ProteinGym_substitutions.zip
rm DMS_ProteinGym_substitutions.zip

# Official CV folds for singles
curl -L -o cv_folds_singles_substitutions.zip "${BASE}/cv_folds_singles_substitutions.zip"
unzip -q cv_folds_singles_substitutions.zip
rm cv_folds_singles_substitutions.zip
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
python featurize.py
```
