# ML_project — gyrB IC50 workflow

These scripts process the checked-in ChEMBL exports and derive descriptor/pIC50 tables. **Ordinary regression includes only positive, finite, exact (`standard_relation == "="`) IC50 values in nM with no non-empty data-validity comment.** Censored `<`/`>` rows, unknown relations, invalid/flagged measurements, other units and missing identity/SMILES are excluded; no censoring-aware model is implemented. Cleaning now retains activity, molecule, assay, document, units, relation and validity provenance and writes reason counts. It does not deduplicate molecule observations: repeated compounds/assays remain and must be grouped when splitting/evaluating. The supplied external collection has overlapping compounds and is not an independent holdout without explicit overlap removal.

## Setup and offline commands

Python 3.10+ and the packages in `requirements.txt` (notably pandas, NumPy, RDKit, scikit-learn and ChEMBL client) are needed. Create an environment and install the manifest with `python -m pip install -r requirements.txt`. Paths are resolved from each script, so commands work from the repository root:

```sh
python scripts/clean_data.py
python scripts/descriptors.py
python scripts/feature_engineering.py
python scripts/pic50_conversion.py
python scripts/clean_external_data.py  # offline, uses checked-in semicolon CSV
python -m unittest discover -s scripts -p 'test_*.py' -v
```

Optional network acquisition only: `python scripts/download_external_data.py` fetches all ChEMBL activity pages for target `CHEMBL2094139`, applies HTTP status/timeout/schema checks, and writes `data/gyrB_external_download.csv` plus a timestamped JSON checksum manifest. It is separate from the checked-in semicolon export consumed by `clean_external_data.py`; this download is not automatically curated or a clean independent test set. `scripts/download_data.py` is another live ChEMBL retrieval path. Live calls are not needed for offline tests.

Model training, hyperparameter tuning and final fitting scripts are separate owner-run research steps, not invoked by this smoke workflow; a final all-data fit is not an evaluation. Existing single random-row splits do not establish scaffold/prospective generalization, and this repository's saved model artifacts are neither required nor loaded by the cleanup/tests. Do not cite model performance without a named untouched, compound/scaffold-aware evaluation set and documented split membership.
