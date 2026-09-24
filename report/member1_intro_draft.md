# Member 1 - Report Draft: Introduction, Problem, Stakeholders, Dataset

*Draft based on the approved dataset proposal. Replace the [brackets] with your own facts and numbers, and rewrite in your own words before submitting.*

## 1. Introduction
Banks reach many customers through direct phone campaigns, but call capacity is limited. Calling every client equally wastes agent time and annoys customers who are unlikely to be interested. This project applies the full data mining workflow - data understanding, preparation, modelling, optimisation and deployment - to predict, **before a call is made**, whether a client will subscribe to a term deposit. The prediction is used to rank clients so that the most promising ones are contacted first.

## 2. Problem definition
- **Task:** supervised binary classification.
- **Target:** `y` - whether the client subscribed to a term deposit (yes / no).
- **Prediction moment:** before the call. Only information known at that moment may be used as input. The call duration (`duration`) is only known after the call ends, so it is excluded to avoid data leakage.
- **Challenge:** the classes are imbalanced (about 11.3% "yes"), so accuracy alone is misleading: predicting "no" for everyone gives about 88.7% accuracy and finds no subscribers. Evaluation therefore focuses on PR-AUC, recall, precision, F1 and lift.
- **Output:** estimated subscription probability, predicted class and a call-priority recommendation based on a documented threshold.

## 3. Stakeholders and requirements
| Stakeholder | Decision supported | Key requirement |
|---|---|---|
| Telemarketing agents | Which clients to call first | Clear ranked list, simple Call / Do not call result |
| Campaign managers | How to use a limited call budget | Adjustable threshold, lift/gain view |
| Bank management | Monitor campaign efficiency | Understandable performance figures |
| Customers | Receive fewer irrelevant calls | Predictions support, not replace, staff decisions; respect opt-outs |

**System inputs:** client profile, loan status, contact channel, campaign history, previous outcome and economic indicators - all available before the call.
**System outputs:** probability, predicted class, priority level.

## 4. Dataset
- **Name / source:** Bank Marketing (with social and economic context), UCI Machine Learning Repository, dataset ID 222. Created by Moro, Rita and Cortez.
- **File used:** `bank-additional-full.csv` - [41,188] records, 20 input attributes and 1 target; May 2008 to November 2010; records are ordered by date.
- **Context:** direct phone marketing campaigns of a Portuguese banking institution.
- **Target distribution:** 36,548 "no" (88.73%) and 4,640 "yes" (11.27%).
- **URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- **Associated paper:** Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22-31.
- **License:** Creative Commons Attribution 4.0.
- **Instructor validation:** [date and evidence - see docs/validation_evidence/].

### Why this dataset fits the scenario
It has a clear, operationally meaningful target; enough records (about 41,000) for training, validation and testing; a mix of categorical and numeric features; class imbalance that justifies imbalance-aware metrics; a documented leakage attribute (`duration`); and date ordering that allows a chronological hold-out.

### Data-quality observations
- Coded missing values: "unknown" in job (330), marital (80), education (1,731), default (8,597), housing (990) and loan (990).
- `pdays = 999` (about 39,673 rows) means "not previously contacted" and is not a real day count.
- 12 exact duplicate rows.
- Strong class imbalance and a temporal shift (the class balance and economic indicators change over time).

### Ethical, privacy and licensing notes
The published data has no direct personal identifiers and is used only for academic work. Predictions could differ across age, job or education groups, so group-level errors should be checked and the system is decision support only. A high probability is not consent to marketing; opt-out and regulatory checks would still be needed. The data comes from one Portuguese bank (2008-2010) and may not transfer to other banks, years or Sri Lankan customers without new validation.

## 5. My own work (Member 1) - to describe in the report
- Client-profile EDA: [key findings for age, job, marital, education].
- Handling of "unknown" values: [decision and evidence from the experiment].
- Age binning and encoding: [decision and evidence].
- Logistic Regression baseline and tuning: [CV PR-AUC baseline vs tuned].
- Threshold tuning: [chosen threshold and reason; lift at the call budget].
- Backend: FastAPI `/predict` endpoint with input validation, and the model export script.
- Home dashboard of the web application.

## 6. Presentation opener - "The business problem and why it matters" (talking points)
1. The bank's agents cannot call everyone; each call costs time and may annoy the customer.
2. Only about 1 in 9 calls ends with a subscription, so most effort is spent on clients who say no.
3. Our system estimates each client's chance before the call and ranks the call list.
4. Example message (fill in your real number): "Calling the top 20% of clients on our list reaches [X]% of all subscribers."
5. It supports agents; people stay in charge of the decision.
