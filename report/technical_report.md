# Technical Report — Term Deposit Call Prioritisation

IT3051 Fundamentals of Data Mining — Mini Project 2026
Bank Marketing (Term Deposit Subscription Prediction)

*Draft assembled from the team's own notebooks, experiment log, and code. Numbers throughout are
pulled directly from `results/experiment_log.csv` and the notebooks — replace any `[bracket]` with
your own words before submitting, but the figures themselves are real, not placeholders.*

---

## 1. Introduction and Problem Definition

Banks reach many customers through direct phone campaigns, but call capacity is limited. Calling
every client equally wastes agent time and annoys customers who are unlikely to be interested.
This project applies the full data mining workflow — data understanding, preparation, modelling,
optimisation and deployment — to predict, **before a call is made**, whether a client will
subscribe to a term deposit. The prediction ranks clients so the most promising ones are contacted
first.

- **Task:** supervised binary classification.
- **Target:** `y` — whether the client subscribed to a term deposit (yes / no).
- **Prediction moment:** before the call. Only information known at that moment may be used. Call
  duration (`duration`) is only known after the call ends, so it is excluded to prevent data
  leakage.
- **Challenge:** the classes are imbalanced (11.27% "yes"), so accuracy alone is misleading —
  predicting "no" for everyone scores 88.73% accuracy while finding zero subscribers. Evaluation
  therefore centres on PR-AUC, recall, precision, F1 and lift.
- **Output:** an estimated subscription probability, a predicted class, and a call-priority
  recommendation based on a documented threshold.

## 2. Assigned Scenario and User/Stakeholder Requirements

| Stakeholder | Decision supported | Key requirement |
|---|---|---|
| Telemarketing agents | Which clients to call first | Clear ranked list, simple Call / Do not call result |
| Campaign managers | How to use a limited call budget | Adjustable threshold, lift/gain view |
| Bank management | Monitor campaign efficiency | Understandable performance figures |
| Customers | Receive fewer irrelevant calls | Predictions support, not replace, staff decisions; respect opt-outs |

**System inputs:** client profile, loan status, contact channel, campaign history, previous
outcome and economic indicators — all available before the call (19 fields total, see Section 6).
**System outputs:** probability, predicted class, priority level (High / Medium / Low), for a
single customer or an uploaded batch.

## 3. Dataset Identification, Source, Citation, and Validation

- **Name / source:** Bank Marketing (with social and economic context), UCI Machine Learning
  Repository, dataset ID 222. Created by Moro, Rita and Cortez.
- **File used:** `bank-additional-full.csv` — 41,188 records, 20 input attributes and 1 target;
  May 2008 to November 2010; records are ordered by date.
- **Target distribution:** 36,548 "no" (88.73%) and 4,640 "yes" (11.27%).
- **URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5K306
- **Associated paper:** Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict
  the success of bank telemarketing. *Decision Support Systems, 62*, 22–31.
- **License:** Creative Commons Attribution 4.0.
- **Instructor validation:** `[date approved, plus the evidence itself — proposal document,
  screenshot, or email — filed in docs/validation_evidence/]`. *This is not something that can be
  reconstructed after the fact; it must be the team's actual record of the approval.*

### Why this dataset fits the scenario
A clear, operationally meaningful target; ~41,000 records for training, validation and testing; a
mix of categorical and numeric features; class imbalance that justifies imbalance-aware metrics; a
documented leakage attribute (`duration`); and date ordering that allows a chronological hold-out.

### Ethical, privacy and licensing notes
The published data has no direct personal identifiers and is used only for academic work.
Predictions could differ across age, job or education groups, so group-level errors should be
checked and the system is decision support only — a high probability is not consent to marketing.
The data comes from one Portuguese bank (2008–2010) and may not transfer to other banks, years, or
markets without new validation.

## 4. Data Understanding and EDA

EDA was split across four notebooks, one per client/feature group, each ending in a table of
"finding → decision" pairs that feed directly into preprocessing.

### 4.1 Client profile (`age`, `job`, `marital`, `education`)
- Age is right-skewed (mean 40.0, median 38, range 17–98); yes-rate is **U-shaped**, not linear —
  65+ converts best (47.2%), <25 is also high (24.0%), while 35–54 is lowest (~8.65%). This is why
  age was binned into `age_group` rather than kept as a raw linear term.
- `job` separates classes strongly: student (31.4%) and retired (25.2%) convert far above average;
  blue-collar (6.9%) and services (8.1%) convert worst.
- "Unknown" values were audited per column rather than blanket-imputed: `job`'s unknown rate
  (11.21%) is indistinguishable from known (11.27%, no signal); `education`'s unknown rate (14.5%
  vs 11.1% known, n=1,731) is a real, consistent gap. Decision: keep "unknown" as its own category
  everywhere rather than imputing — a mode-imputation experiment later gave no real PR-AUC gain
  (0.2025 vs 0.2027, inside the ±0.025 CV standard deviation).
- Cramer's V ranking of the four features: age (binned) 0.171 > job 0.153 > education 0.068 >
  marital 0.055 — age and job carry most of the client-profile signal.

### 4.2 Target, imbalance, credit features, and leakage
- Target imbalance confirmed: 88.73% "no" / 11.27% "yes" across 41,188 clients.
- `default` has 8,597 "unknown" values and only 3 clients with `default = yes` (none subscribed) —
  very little information content.
- `housing` and `loan` have similar subscription rates between categories and 990 "unknown" values
  each; "unknown" retained as its own category.
- **Duration leakage, quantified:** clients who subscribed had a far longer average call
  (553.19s) than those who didn't (220.84s); median rose from 163.5s ("no") to 449s ("yes").
  Since duration is only known *after* the call, using it would leak the outcome and produce
  unrealistically high — and useless for deployment — model performance. `duration` is excluded
  from every model in this project.
- 12 exact duplicate rows were identified and removed (41,188 → 41,176), before `duration` was
  dropped (see Section 5 for why the order matters).

### 4.3 Campaign history (`contact`, `poutcome`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`)
- `poutcome`: a previous **success** converts at 65.1% vs 8.8% for "never contacted" — about 5.8×
  the overall rate, the single strongest categorical signal found in the project, though on a
  narrow group (3.3% of clients).
- `month`: May holds a third of all calls (33.4%) at the *lowest* rate (6.4%); March, September,
  October and December convert at 44–51% but are only 4.9% of calls combined.
- `contact`: cellular converts at 14.7% vs telephone at 5.2% — flagged as a pattern worth noting
  for the business, but not claimed as causal, since it is partly confounded with time (the oldest
  fifth of the file is 100% landline).
- `day_of_week`: rates range only 10.0%–12.1% across the week — a genuinely weak feature.
- Cramer's V strength ranking: `poutcome` 0.32 > `month` 0.28 > `contact` 0.15 >> `day_of_week`
  0.025 — `day_of_week` was flagged as the first candidate to drop in feature selection (see 4.4 /
  Section 6, where this was tested and the drop did **not** help).

### 4.4 Macroeconomic indicators (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`)
- Strong multicollinearity: `emp.var.rate` ↔ `euribor3m` r=0.97, `euribor3m` ↔ `nr.employed`
  r=0.95, `emp.var.rate` ↔ `nr.employed` r=0.91.
- Relationship with the target: `nr.employed` r=−0.355, `euribor3m` r=−0.308, `emp.var.rate`
  r=−0.298 (negative — subscriptions more common when employment/rates were lower);
  `cons.conf.idx` is only weakly related (r=0.055).
- A SelectKBest (mutual information) demonstration ranked `euribor3m`, `cons.conf.idx`, and
  `cons.price.idx` as the top three — used as an exploratory signal, not a final decision (final
  feature-selection testing is in Section 6).

## 5. Data Cleaning and Preprocessing

- **Duplicate order matters.** `duration` must be dropped *after* duplicates are removed, not
  before — the team found that dropping `duration` first made 1,784 genuinely different calls look
  identical and deleted them by mistake. The shared `remove_duplicates()` helper in
  `src/data_cleaning.py` enforces this order; after the fix, exactly 12 true duplicates are
  removed (41,188 → 41,176), matching the number stated in the original dataset proposal.
- **Leakage prevention.** `duration` is excluded from every model's input set, leaving 19
  predictor features. This is the project's central leakage-prevention decision and is enforced
  in code: `backend/schemas.py`'s `ClientFeatures` has `extra="forbid"`, so even an API client that
  tries to submit `duration` is rejected.
- **Missing/"unknown" values.** Coded as the string `"unknown"` rather than true nulls, present in
  `job` (330), `marital` (80), `education` (1,731), `default` (8,597), `housing` (990), `loan`
  (990). Rather than imputing, "unknown" is kept as its own category across all of these — tested
  against mode-imputation and found to perform no better (within CV noise), while retaining real
  signal in columns like `education` (Section 4.1).
- **`pdays = 999`** is a *code* meaning "never contacted before", not a real day count. It is
  recoded to `-1` so models don't treat it as an enormous number of days, and a separate boolean
  `pdays_known` flag is derived instead (see `CampaignFeatures`, Section 6).
- **Train/test split.** The dataset is ordered by date, so a **chronological 80/20 hold-out**
  (train on the earliest ~80%, test on the most recent ~20%) was used as the primary split across
  all four models — this is what a real deployment looks like (predicting future campaigns from
  past data). A stratified random 80/20 split was also tested as a secondary comparison. The
  chronological split revealed a major finding: **the training period has a 6.38% positive rate,
  but the test period has 30.83%** — a ~5× shift, invisible under a random split. This shift is
  discussed throughout Sections 8–9 and is the project's central limitation (Section 14).
- **Scaling.** Numerical features are standardised inside the pipeline (fit on training data only)
  for the two scale-sensitive algorithms, Logistic Regression and SVM. Tree-based models
  (Random Forest, XGBoost) do not require scaling.
- **Encoding.** All categorical features are one-hot encoded with `handle_unknown="ignore"`, so an
  unseen category at inference time doesn't crash the API — it's encoded as all-zero instead.
- **Imbalance handling** differs by model family, chosen for each algorithm rather than applied
  uniformly: `class_weight="balanced"` for Logistic Regression and SVM; `scale_pos_weight` for
  XGBoost (14.69 for the chronological split, versus 7.88 for the stratified split — must always
  be computed from the *training* labels of the split actually being used); and an explicit
  three-way comparison (no handling vs. class weights vs. SMOTE) for Random Forest, which selected
  class weighting (Section 9).

## 6. Feature Engineering and Feature Selection

Two custom scikit-learn transformers were built so the exact same feature logic runs identically
in training, in the model-comparison notebooks, and in the live backend — one pipeline, no
re-implementation:

- **`CampaignFeatures`** (`src/features.py`) — engineers the campaign/previous-contact columns:
  - `previously_contacted` = 1 if `previous` > 0, else 0. This is used instead of `pdays == 999`
    to detect "never contacted before", because `pdays = 999` does **not** reliably mean that in
    the raw data — some clients with `pdays = 999` have `poutcome = 'failure'`, meaning they
    *were* contacted but the gap wasn't recorded. `previous > 0` matches `poutcome` exactly
    (verified in `notebooks/03_eda_campaign.ipynb`, Section 6).
  - `pdays_known` = 1 if `pdays` holds a real day count, 0 if it holds the 999 code.
  - `pdays` recoded from 999 to −1 (a code, not a day count).
  - `campaign` capped at a high quantile (default 99th percentile) **learned from training data
    only** inside `fit()`, so no test-set information leaks into the cap — outliers are capped,
    not removed, since they represent real customers, not data errors.
- **`UnknownHandler` / `AgeGrouper`** (`src/client_features.py`) — implement the "unknown"
  retention and age-binning decisions from Section 4.1/5 as reusable, fit/transform-safe steps.

### Feature selection: tested, not assumed
Rather than dropping features on correlation/Cramer's V alone, the team ran a controlled
experiment (SVM notebook, Section 3): remove `emp.var.rate`, `nr.employed`, and `day_of_week` (the
three features flagged as redundant or weak in EDA) and compare the same model under the same
cross-validation folds. **Result: the reduced 16-feature model scored a lower PR-AUC (0.1429) than
the full 19-feature model (0.1563)** — removing them made performance worse, not better, despite
the EDA correlation/Cramer's V evidence suggesting they were redundant. All 19 features were
therefore retained for every model. This is reported as a genuine, evidence-based negative result,
not reframed as a success — it demonstrates why feature-selection decisions must be validated
experimentally rather than taken directly from exploratory statistics.

## 7. Algorithms Implemented and Their Rationale

Four algorithms spanning distinct model families were implemented, to satisfy the requirement of
comparing genuinely different approaches rather than near-duplicates:

| Algorithm | Owner | Family | Why chosen |
|---|---|---|---|
| Logistic Regression | Member 1 | Linear | Interpretable baseline every other model must beat; coefficients convert directly into odds ratios the bank can act on |
| Random Forest | Member 2 | Bagging ensemble (trees) | Captures non-linear interactions without scaling; natural fit for mixed categorical/numeric data |
| XGBoost | Member 3 | Boosting ensemble (trees) | Strong general-purpose classifier; supports `scale_pos_weight` for imbalance natively; two tuning strategies compared |
| SVM (RBF kernel) | Member 4 | Margin-based / kernel | Different inductive bias again (maximum-margin, kernel trick); tests whether a non-tree non-linear approach adds value |

## 8. Model Evaluation and Comparison

**Validation strategy (shared by all four models):** chronological 80/20 hold-out as the primary
evaluation (train on the earlier ~80% of rows, test on the most recent ~20%), with stratified
5-fold cross-validation inside the training portion for model selection and tuning. All four
models use the *same* `chronological_split()` function and the same `random_state=42`, so the
train/test boundary is identical across models and the comparison is fair. The test set is used
exactly once per model, at the end.

**Primary comparison (chronological test set, sorted by Test PR-AUC):**

| Model | Test PR-AUC | Test ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.5052** | 0.7007 | 0.8476 | 0.3599 | 0.5053 |
| XGBoost (tuned) | 0.4603 | 0.6685 | 0.5719 | 0.4430 | 0.4992 |
| Random Forest (tuned) | 0.4459 | 0.6536 | 0.4777 | 0.4259 | 0.4503 |
| SVM (tuned) | 0.2743 | 0.4872 | 0.8488 | 0.3429 | 0.4884 |

*(Random-classifier floor on this test set ≈ 0.308, i.e. the test-period positive rate.)*

**A critical, honest finding that applies to all four models, not just the weaker ones:** an
internal forward-looking validation block (train on the first 75% of the training period, validate
on the last 25%, still never touching the real test set) showed **every model — including Logistic
Regression — at chance level** (ROC-AUC 0.49–0.54, PR-AUC near the local positive-rate floor). The
macroeconomic indicators barely overlap in range between the training and test periods (e.g.
`euribor3m` moves from 4.2–5.0 in the earliest rows to 0.6–1.3 in the test period), so they behave
almost like a clock. Tree-based models cannot predict outside the value ranges they saw in
training, while a linear model can extrapolate — a plausible structural reason Logistic Regression
holds up comparatively well on the real chronological test set. This is discussed further as the
project's central limitation in Section 14.

A secondary stratified-split comparison was also logged (not directly comparable to the table
above, since a random split mixes early/late rows into both train and test and therefore has a
different test-set class balance): under a stratified split, XGBoost (0.4536) and Logistic
Regression (0.4467 baseline) were statistically level — demonstrating that the chronological
split, not the model, is responsible for most of the separation seen in the primary table.

## 9. Hyperparameter Tuning and Optimization

| Model | Method | CV PR-AUC: baseline → tuned | Test PR-AUC: baseline → tuned |
|---|---|---|---|
| Logistic Regression | GridSearchCV (`C` ∈ [0.001, 100], L1/L2) | 0.2027 → 0.2027 (no config beat baseline by >0.0022, inside ±0.025 CV std) | 0.5052 → 0.5052 (identical — tuning picked the same config as baseline) |
| Random Forest | 3-way imbalance comparison (none / class-weight / SMOTE) → RandomizedSearchCV on the winner | 0.1589 (best pre-tuning) → 0.2166 | — → 0.4459 |
| XGBoost | RandomizedSearchCV (40 iters) **vs.** Optuna (40 trials), same search space, same folds | 0.1805 → 0.2232 (Random) / 0.2236 (Optuna) | 0.367 → 0.460 |
| SVM | Feature-selection experiment (19 vs. 16 features) → GridSearchCV (`C`, `gamma`) on the winner | 0.1563 → 0.1773 | — → 0.2743 |

**Member 3's optimisation-topic finding (XGBoost, RandomizedSearchCV vs. Optuna):** the two search
strategies reached statistically indistinguishable CV scores (0.2232 vs. 0.2236 — a difference of
0.0004, far smaller than the fold-to-fold standard deviation of ≈0.014), in roughly the same wall
-clock time (~40 seconds each for 40 trials). This is reported as a genuine tie, not a win for
Optuna — its sequential, informed-search advantage needs either a larger trial budget or a harder
search space to show up, and with only 40 trials on this 8-parameter space it had not had room to
do so. Both curves climb quickly in the first ~10 trials then flatten, consistent with a search
space that is not very large to begin with.

**Random Forest's optimisation-topic finding (Member 2):** among the three imbalance strategies,
plain class weighting was selected over SMOTE — SMOTE achieved the highest raw recall (0.0976 in
CV) but at a cost to PR-AUC and precision, and it also avoids generating synthetic *categorical*
observations (a known weakness of naive SMOTE on mixed-type data), which class weighting avoids
entirely.

**SVM's optimisation-topic finding (Member 4):** the feature-selection experiment (Section 6)
showed the full 19-feature set outperforming the reduced set, so tuning (`C`, `gamma`, RBF kernel)
was run on all 19 features; the best configuration (`C=0.1`, `gamma=scale`) improved CV PR-AUC from
0.1563 to 0.1773 and delivered the highest recall of any model in the project on the final test set
(0.8488), at the cost of precision (0.3429).

**Logistic Regression's optimisation-topic finding (Member 1):** a dedicated threshold-tuning
study (on out-of-fold training predictions, never the test set) found the best-F1 operating
threshold at **0.720**. A second candidate — a threshold calibrated to call exactly the top 20% of
a ranked list — was also computed (0.526), but was found to be unreliable: because the test
period's probability distribution shifts upward relative to training (Section 8), that
training-calibrated threshold actually calls 66.4% of test-period clients in practice, not 20%.
This mismatch is reported honestly as a deployment limitation (Section 14), not hidden.

## 10. Final Model Selection and Justification

**Final model: Logistic Regression** (`C=1`, L2 penalty, `class_weight="balanced"`), deployed with
the best-F1 call threshold of **0.72**.

**Justification, built from the comparison table in Section 8:**
1. **Highest chronological test PR-AUC** among all four models (0.5052), by a clear margin over
   the second-best tuned model (XGBoost, 0.4603).
2. **Interpretability** — Logistic Regression's coefficients translate directly into odds ratios
   the bank can explain to non-technical stakeholders (e.g. "a previous successful contact
   multiplies a client's odds of subscribing by X"), unlike the ensemble/kernel alternatives.
3. **Only model with a completed, documented threshold-tuning step** (Section 9), making its
   call-priority recommendation genuinely deployment-ready rather than using an arbitrary 0.5 cut
   -off.
4. **Fastest to train and serve** in the live API — relevant for a system that may need periodic
   retraining as new campaign data arrives (Section 14).

This is a deliberate trade-off, not an oversight: the tuned SVM achieves the **highest recall of
any model** in the project (0.8488, finding 85% of true subscribers), but its much lower precision
(0.3429) and PR-AUC (0.2743) place it last overall — it was not selected, because a system that
flags the majority of all clients as "call" at a 34% hit rate does not meaningfully help an agent
prioritise. A possible explanation for the ranking overall: the test period shows a strong
distribution shift from the training period (Section 8), and a well-regularised linear model
generalises more robustly across that shift than models tuned primarily to the training-period
distribution — consistent with the forward-looking validation-block finding that every model
collapses toward chance level under this shift. This selection was made by the whole team in the
combined comparison notebook (`notebooks/10_model_comparison_final.ipynb`), not by any one member.

## 11. System Architecture and Implementation

```
Raw CSV (bank-additional-full.csv)
        │
        ▼
src/data_cleaning.py   — load, dedupe (before dropping duration), chronological/stratified split
        │
        ▼
CampaignFeatures + UnknownHandler + AgeGrouper   — feature engineering (src/features.py, src/client_features.py)
        │
        ▼
ColumnTransformer   — one-hot encode categoricals, scale numerics (model-dependent)
        │
        ▼
Classifier   — Logistic Regression (final), or RF / XGBoost / SVM (compared, not deployed)
        │
        ▼
One sklearn Pipeline, trained once, exported as models/final_model.joblib
   (bundles: the fitted pipeline + model_name + deployed threshold + input_columns)
        │
        ▼
backend/app.py (FastAPI)   — loads the bundle once at startup, exposes REST endpoints
        │
        ├──▶ frontend-web/ (Next.js, TypeScript, Tailwind)   — production UI
        └──▶ frontend/ (Streamlit)   — earlier working UI, kept functional
```

Because preprocessing and the model are **one fitted pipeline**, the backend never re-implements
feature engineering — it calls the exact same `.predict_proba()` that training called, which is
what Stage 9's requirement ("apply the same preprocessing used during model development") means in
practice here, not just in principle.

**Technology stack:** Python (pandas, scikit-learn, XGBoost, Optuna, joblib) for the data/model
side; FastAPI + Pydantic for the backend; Next.js (App Router), TypeScript and Tailwind CSS for the
production frontend, with Streamlit as an earlier working alternative. Full dependency list in
`requirements.txt`.

## 12. Backend and Frontend Development

### Backend (`backend/app.py`, `backend/schemas.py`)
Five REST endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + whether a model is currently loaded |
| `GET /model-info` | Deployed model name, call threshold, expected input columns |
| `POST /predict` | Score one client, return probability + recommendation + priority |
| `GET /predict-batch/template` | Download a blank CSV with the correct 19 (+ optional `client_id`) headers |
| `POST /predict-batch` | Upload a CSV of clients, return them ranked by probability |

Input validation uses Pydantic (`ClientFeatures`): every field has the exact bounds enforced during
training (e.g. `age` 17–100, `campaign` 1–60), categorical fields are restricted to the literal
values seen in training, and `extra="forbid"` means a client that tries to submit `duration` is
rejected outright rather than silently ignored — the leakage prevention from Section 5 is enforced
at the API boundary too, not only in the notebooks. Batch upload validates in three stages: file
type (CSV only), required columns present, and per-row field validation — with a `/predict-batch`
response that names exactly which rows and columns failed, and why, rather than a generic error.

### Frontend (`frontend-web/`, Next.js)
Four pages, each built against every real interaction state (not just the success path):

- **Home** — system status (API reachable? model loaded? current threshold), model performance
  summary pulled from `results/experiment_log.csv`, and an offline/unavailable state when the API
  can't be reached.
- **Single Prediction** — the full 19-field form, grouped to match the feature groups in Section
  4/6 (client profile, credit/loans, contact, campaign history, economic context); client-side
  range validation using the exact same bounds as the backend; loading and error states.
- **Batch Upload** — drag-and-drop CSV upload, a downloadable template, three distinct validation
  error states (wrong file type, missing columns, row-level errors shown in a table with the
  actual invalid value and a plain-language fix), and a ranked, filterable, downloadable call list
  on success.
- **Insights** — the model-comparison chart from Section 8 (built from the live experiment log,
  not a static image), the deployed model's details, and an empty-data state.

An earlier Streamlit app (`frontend/`) remains functional as a secondary interface to the same
backend.

## 13. System Testing and Results

**74 automated tests, all passing**, across nine files:

- Unit tests for data cleaning, feature transformers (`CampaignFeatures`, `UnknownHandler`,
  `AgeGrouper`), and each model's pipeline-building functions (Logistic Regression, Random Forest,
  XGBoost, SVM) — including edge cases like an empty/all-missing `campaign` column, nullable
  integer dtypes, and calling `.transform()` before `.fit()`.
- 14 backend API integration tests (`backend/tests/test_api.py`), covering both success and
  failure paths: correct predictions and model-info; `503` when no model is loaded; `422` on
  out-of-range values; rejection of a submitted `duration` field; CSV template correctness;
  batch ranking order; rejection of non-CSV files, missing columns, and empty files; and row-level
  validation error reporting with the correct CSV row number (accounting for the header row).

**Manual end-to-end verification:** every frontend screen and error state described in Section 12
was exercised against the live backend with real HTTP requests (not mocked), including uploading
CSVs with deliberately invalid rows to confirm the error messages shown to the user match what the
API actually returns.

## 14. Limitations and Possible Future Improvements

### Limitations (found and documented through the project, not assumed)
1. **Temporal distribution shift.** The training period's positive rate (6.38%) is roughly a
   quarter of the test period's (30.83%). An internal forward-looking validation block showed
   *every* model collapsing to near-chance performance (ROC-AUC 0.49–0.54) under this shift — this
   is a property of the dataset's time ordering, not a weakness specific to any one algorithm.
2. **Macroeconomic features barely overlap between periods** (e.g. `euribor3m` ranges are almost
   disjoint across the earliest, middle, and test thirds of the file), which likely explains why
   tree-based models (which cannot extrapolate beyond training value ranges) underperform the
   linear model on this particular test split.
3. **A fixed call threshold does not transfer reliably across the shift** — a threshold calibrated
   to call 20% of clients on training-period data calls 66.4% of clients in the test period in
   practice (Section 9). A real deployment would need periodic threshold recalibration against
   recent data, not a single fixed cut-off.
4. **Precision at usable recall is limited** — even the final model's precision (0.36) means most
   "Call" recommendations will not convert; the system is explicitly decision support, not
   automation, and is presented to users that way throughout the frontend.
5. **Associations found in EDA (Section 4) are correlational, not causal** — e.g. the cellular-
   vs-telephone gap is partly confounded with calling period, and this is stated directly in the
   EDA rather than implied as causation.
6. **Single-source, single-era dataset.** Data comes from one Portuguese bank, 2008–2010; applying
   the model to a different bank, country, or era without revalidation is not supported by this
   evidence.

### Future improvements
- Periodic retraining/recalibration against recent campaign data, with monitoring for distribution
  drift (e.g. tracking the macroeconomic feature ranges seen in production against training).
- A rolling-window or time-aware validation strategy, rather than a single chronological split, to
  better estimate how performance degrades over time.
- A/B testing the deployed threshold against real call outcomes, rather than relying solely on a
  training-period calibration.
- Richer behavioural/contact-history features if the bank's systems can supply them (e.g. days
  since last *any* contact, not just previous-campaign contact).
- Extending the frontend's Insights page with real (not illustrative) outreach findings once the
  team runs the dedicated EDA needed to validate them.

## 15. Individual/Group Contributions

| Member | Main contributions |
|---|---|
| Member 1 | Client-profile EDA; `UnknownHandler` / `AgeGrouper` feature transformers; Logistic Regression model (baseline, tuning, threshold study — final selected model); backend `/predict` endpoint and model export script; original Home dashboard |
| Member 2 | Target/imbalance/leakage EDA; duplicate-removal and chronological-split groundwork; Random Forest model (imbalance-strategy comparison, tuning, final evaluation) |
| Member 3 | Campaign-history EDA; `CampaignFeatures` transformer; XGBoost model (RandomizedSearchCV vs. Optuna comparison, tuning, final evaluation); `/predict-batch` backend endpoint and its validation; full Next.js production frontend (all four pages, every interaction/error state) |
| Member 4 | Macroeconomic EDA and multicollinearity analysis; `src/preprocessing.py` numeric utilities; SVM model (feature-selection experiment, tuning, final evaluation) |
| Group | Dataset proposal and instructor validation; shared experiment log (`results/experiment_log.csv`); combined model-comparison notebook and final model selection; this report |

*(This table is built from notebook authorship headers and commit history. Each member should
check and adjust their own row before submission.)*
