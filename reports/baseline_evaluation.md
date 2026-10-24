# Enpres.x Baseline Evaluation

## Scope

Selected buildings: Panther_lodging_Kirk, Panther_office_Clementine, Panther_education_Misty, Panther_education_Cleopatra, Panther_education_Enriqueta

Rows loaded for selected buildings: **87,720**
Missing electricity targets: **16,180 (18.45%)**

Missing electricity targets were excluded from training and scoring. They were never replaced with zero or interpolated.

## Chronological split

- Training: through `2017-05-26 15:00:00` (**44,242** usable rows)
- Validation: after training through `2017-09-13 07:00:00` (**13,086** usable rows)
- Test: after validation through `2017-12-31 23:00:00` (**13,118** usable rows)

Usable rows require an observed target and an observed same-building value 24 hours earlier. Weather feature gaps were handled by a training-fitted median imputer.

## Test results

Both methods were evaluated on exactly the same held-out test rows.

| Method | MAE | RMSE | Rows |
|---|---:|---:|---:|
| Model | 4.1253 | 6.9193 | 13,118 |
| Same-hour previous-day benchmark | 7.3050 | 12.3879 | 13,118 |

## Validation results

| Method | MAE | RMSE | Rows |
|---|---:|---:|---:|
| Model | 3.0786 | 4.8498 | 13,086 |
| Same-hour previous-day benchmark | 5.5733 | 9.2246 | 13,086 |

## Saved outputs

- `models/panther_baseline.joblib`
- `reports/baseline_metrics.json`
- `reports/baseline_test_predictions.csv`

The saved model is a scikit-learn pipeline containing training-fitted preprocessing and a histogram gradient-boosting regressor. The benchmark is not a trained model; it uses the observed lag-24 value.
