# Bank Marketing – Term Deposit Subscription Prediction

IT3051 Fundamentals of Data Mining – Mini Project 2026.
Predicts, before a call is made, the probability that a client subscribes to a term deposit, so telemarketing staff can prioritise calls.

- Dataset: UCI Bank Marketing (ID 222), `bank-additional-full.csv` – place it in `data/raw/`
- Citation: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5K306
- Key decision: `duration` is excluded from the deployable model (leakage).

## Layout
| Folder | Purpose |
|---|---|
| `data/` | raw (never edited), interim, processed |
| `docs/` | dataset proposal and validation evidence |
| `notebooks/` | EDA (01–04), preprocessing (05), models (06–09), comparison (10) |
| `src/` | shared code: config, cleaning, features, preprocessing, train, evaluate |
| `models/` | `final_model.joblib` (full pipeline) and per-member experiments |
| `results/` | figures and `experiment_log.csv` |
| `backend/` | FastAPI service and tests |
| `frontend/` | Streamlit app |
| `report/`, `presentation/` | technical report and slides |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
