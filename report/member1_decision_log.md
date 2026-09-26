# Member 1 — Decision Log (PE1)

| Decision | Evidence | Alternative rejected |
|---|---|---|
| Keep 'unknown' as its own category (job, marital, education) | education: unknown yes-rate 14.5% vs known 11.1% (real signal, n=1,731); mode imputation gave no CV PR-AUC gain (0.2025 vs 0.2027, within ±0.025 std) | Mode imputation |
| Bin age into age_group | Yes-rate is U-shaped by age (65+: 47.2%, <25: 24.0%, 35-54: ~8.65%) — not linear; age (binned) has the strongest Cramer's V (0.171) of all client-profile features | Raw age as a linear numeric term |
| One-hot encode job/marital/education/age_group with handle_unknown='ignore' | Nominal categories, no natural order; 'ignore' prevents API crashes on unseen categories | Ordinal encoding |
| Chronological hold-out (last 20%) as primary split | Train yes-rate 6.38% vs test 30.83% — confirms real temporal shift the proposal warned about (Section 7) | Random stratified split (hides the shift) |
| Keep LR baseline hyperparameters (C=1, L2) | Grid search across C in [0.001,100] and both penalties found nothing that beat baseline by more than 0.0022 PR-AUC (within ±0.025 CV std) | Any tuned configuration |
| Fixed 20%-call-budget threshold flagged as a limitation | Threshold (0.526) computed on train-period data calls 66.4% of test-period clients instead of 20%, due to the temporal shift in class balance | Treating the threshold as fixed/reliable without noting the shift |

## Bug found and fixed
Original `load_data()` dropped `duration` before checking duplicates, which incorrectly merged 1,784 rows with different call lengths into "duplicates." Fixed by deduplicating first (using the team's `remove_duplicates()`), then dropping `duration` — now matches the proposal's 12 duplicates exactly (41,188 → 41,176 rows).