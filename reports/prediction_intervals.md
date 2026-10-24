# Enpres.x Empirical 80% Prediction Intervals

## Method

The interval uses a symmetric absolute-residual quantile calibrated only on the validation period.
For each prediction, the bounds are:

```text
lower = max(0, model_prediction - q80_absolute_validation_residual)
upper = model_prediction + q80_absolute_validation_residual
```

- Nominal target coverage: **80%**
- Validation rows used for calibration: **13,086**
- Validation period: after `2017-05-26 15:00:00` through `2017-09-13 07:00:00`
- Calibrated half-width: **4.9706** consumption units

The test actual values were not used to construct or tune the interval. The lower bound was clipped at zero because electricity consumption cannot be negative.

## Held-out test coverage

- Test period: `2017-09-13 08:00:00` through `2017-12-31 23:00:00`
- Observed test rows scored: **13,118**
- Overall empirical coverage: **72.91%**

80% is the nominal target based on validation calibration. Actual held-out test coverage can be higher or lower because the test period may have a different error distribution.

## Coverage by selected building

| Building | Test rows | Coverage |
|---|---:|---:|
| Panther_lodging_Kirk | 2,632 | 48.71% |
| Panther_office_Clementine | 2,632 | 87.50% |
| Panther_education_Misty | 2,616 | 91.21% |
| Panther_education_Cleopatra | 2,632 | 77.51% |
| Panther_education_Enriqueta | 2,606 | 59.59% |

## Saved outputs

- `reports/baseline_test_predictions_with_intervals.csv`
- `reports/prediction_interval_metrics.json`

The original `reports/baseline_test_predictions.csv` and baseline evaluation report were preserved unchanged.
