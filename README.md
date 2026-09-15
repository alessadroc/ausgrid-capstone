# Electricity Demand Forecasting from Ausgrid Substation Data

Capstone research project — 41029 Engineering Research Preparation, UTS.

**Research question:** How can machine learning models, using multi-year
substation load data, improve electricity demand forecasting and thus
reliability in urban distribution networks?

## Data

Ausgrid distribution zone substation data, publicly released:
https://ausgrid.com.au/about-us/about-ausgrid/research-data-sets/distribution-zone-substation-data

Download the CSV exports into `data/raw/`. The raw data is **not** committed
to this repository — it is large and is redistributed under Ausgrid's terms.

Study period: 2023-01-01 to 2025-12-31. Earlier years are excluded because
COVID-19 lockdown periods (2020–2022) distorted the residential/commercial
demand split in ways that do not recur, which would degrade a model evaluated
on post-pandemic conditions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Inspect the raw archive (seconds — reads headers only)
python scripts/inspect_data.py --input data/raw

# 2. Preprocess to an hourly modelling dataset
python scripts/run_preprocessing.py \
    --input data/raw --output data/processed \
    --start-date 2023-01-01 --end-date 2025-12-31 --stream

# 3. Train the three architectures
python scripts/train_baseline.py --config configs/arima.yaml
python scripts/train_deep.py --config configs/lstm.yaml
python scripts/train_deep.py --config configs/transformer.yaml

# 4. Evaluate and produce comparison tables
python scripts/evaluate.py --results results/metrics
```

## Repository layout

| Path | Contents |
|---|---|
| `src/ausgrid_pipeline/` | Preprocessing package (ingestion, cleaning, features) |
| `src/models/` | ARIMA, LSTM, Transformer implementations |
| `src/evaluation/` | Error metrics and proxy reliability metrics |
| `scripts/` | Command-line entry points |
| `configs/` | Experiment configurations (one file per model run) |
| `notebooks/` | Exploratory analysis — not part of the reproducible pipeline |
| `results/` | Metrics, figures, trained models |
| `tests/` | Correctness checks for the preprocessing logic |
| `docs/` | Methodology notes, data dictionary |

## Method summary

Three architectures compared under identical data conditions:

| Model | Role |
|---|---|
| ARIMA | Statistical baseline |
| LSTM | Recurrent deep learning |
| Transformer | Attention-based deep learning |

Forecast horizons: 1 hour (real-time operational) and 24 hours (day-ahead
planning).

Evaluation: MAE, RMSE and MAPE, plus three proxy reliability metrics — peak
load prediction error during high-demand periods, load balancing deviation
across substations, and forecast bias frequency during peak events.

## Notes

Train/validation/test splits are strictly chronological. Scaling parameters are
fitted on the training block only. Imputed and outlier-flagged observations are
excluded from evaluation targets.
