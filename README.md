# ING Türkiye Datathon — ML pipeline archive

A competition/research archive of customer-behaviour modeling experiments:
temporal feature engineering, LightGBM/XGBoost/CatBoost and classical models,
Optuna tuning and ensemble blending. It is a collection of experiments, not a
deployed prediction service.

## Two experiment paths

| Entry point | Inputs / outputs |
|---|---|
| `data_exploration.py` | Reads raw competition CSVs and produces exploratory summaries/plots |
| `feature_engineering.py` | Writes `engineered_features.csv`; an earlier feature path |
| `feature_engineering_optimized.py` | Writes `train_features.csv` and `test_features.csv` |
| `ensemble_model.py` | Reads the train/test feature pair and writes ensemble/individual submissions |
| `hyperparameter_optimization.py` | Uses the feature pair for Optuna experiments and optimized submissions |
| `inghub11.py`, `inghub11.ipynb` | Separate integrated experiment with its own configuration/Colab paths |

The earlier `engineered_features.csv` output is not automatically the input
contract for the ensemble scripts. Inspect target columns and data preparation
before combining experiment paths.

## Environment and data

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Use only competition data you are permitted to access. Scripts expect filenames
such as `customers.csv`, `customer_history.csv`, `referance_data.csv`,
`referance_data_test.csv` and, for exploration, `sample_submission.csv`. The
`referance` spelling is intentional to match the source code. Data and generated
submissions are excluded. Some integrated paths assume Colab/Drive locations;
edit the corresponding configuration for your local authorized data.

For the optimized path, after verifying your input schema:

```bash
python feature_engineering_optimized.py
python ensemble_model.py
# Optional, computationally expensive: python hyperparameter_optimization.py
```

## Validation and limitations

Python sources were syntax-checked during the audit. No competition data, model
training, Optuna trial or end-to-end submission was run, and no new performance
or leaderboard claim is made. Notebook outputs are historical observations.
There is no automated regression suite or pinned reproducible environment yet.

Temporal reference dates, customer overlap and feature cutoffs matter for leakage.
Cross-validation in the code does not by itself establish a leakage-free evaluation.
Preserve raw data rights, document splits/seeds with future runs, and keep customer
records and identifiers out of Git and public screenshots.


## License and contact

No license file is currently included; no new license or redistribution grant is asserted here. Author: [Muhammet Fatih Çetintaş](https://github.com/M-Fatih-C).
