# Enpres.x Phase 2 Report

## Scope

Phase 2 adds three practical, demo-ready extensions to the existing Panther baseline without retraining the model, changing raw data, or overwriting the original evaluation outputs:

1. Meter data-quality flags.
2. Potential audit-candidate ranking.
3. Separate illustrative 10% and 8% retrofit scenarios.

The optional unseen-site evaluation was not attempted because it would require a separate leakage-safe training/evaluation workflow and was not necessary to complete the first three priorities reliably.

## 1. Meter data-quality flags

The dashboard now reports missing readings, long missing gaps, suspicious spikes, and confirmed faulty readings separately.

Rules:

- A missing reading is a null or non-numeric electricity target.
- A long missing gap is at least 24 consecutive missing hourly readings for one building.
- A suspicious spike is above 3× the centered 24-hour rolling median and more than 6 rolling median absolute deviations above that baseline.
- No confirmed faulty readings are asserted because the project has no meter-maintenance or validation labels.

Current selected-building results:

- Prepared rows: **87,720**
- Missing readings: **16,180**
- Long-gap runs: **245**
- Readings inside long gaps: **10,660**
- Suspicious spikes: **1**
- Confirmed faulty readings: **0**
- Longest selected-building gap: **144 hours**

Flags are descriptive only. Raw readings remain untouched, and missing targets are not filled with zero.

## 2. Potential audit-candidate ranking

The ranking uses the saved held-out predictions and observed test electricity for the five baseline buildings. Residual is:

```text
observed electricity - saved model prediction
```

Eligibility requires at least 168 observed test hours and a positive residual streak of at least 6 hours. The score is:

```text
sustained positive-residual hours × mean positive residual
```

The current ranking is:

| Rank | Building | Sustained positive hours | Mean positive residual | Screening score |
|---:|---|---:|---:|---:|
| 1 | Panther_education_Enriqueta | 683 | 6.41 | 4380.4 |
| 2 | Panther_education_Cleopatra | 358 | 3.35 | 1200.7 |
| 3 | Panther_lodging_Kirk | 159 | 6.53 | 1038.1 |
| 4 | Panther_education_Misty | 340 | 2.10 | 712.6 |
| 5 | Panther_office_Clementine | 167 | 2.72 | 453.6 |

These are investigation priorities, not findings of energy waste. Positive residuals can reflect weather mismatch, occupancy, model misspecification, or data-quality problems.

## 3. Illustrative retrofit scenarios

The dashboard applies two separate hypothetical reductions to observed held-out test-period electricity for the selected building:

```text
estimated energy reduction = observed baseline energy × assumed percentage
```

For example, for `Panther_lodging_Kirk`:

| Scenario | Observed baseline energy | Assumed reduction | Estimated reduction | Scenario energy |
|---|---:|---:|---:|---:|
| Illustrative 10% reduction | 416,012.08 | 10% | 41,601.21 | 374,410.87 |
| Illustrative 8% reduction | 416,012.08 | 8% | 33,280.97 | 382,731.11 |

The scenarios are alternatives, not cumulative measures. They are illustrative energy estimates only, not measured or guaranteed savings. No financial savings are claimed.

## Safeguards preserved

- `models/panther_baseline.joblib` was preserved.
- Original baseline and interval reports were preserved.
- Raw datasets were not modified.
- Missing electricity targets were not filled with zero.
- The guarded live 24-hour forecast still requires valid future weather and remains unavailable without it.

## Files

- `src/phase2_extensions.py`
- `tests/test_phase2_extensions.py`
- `app/streamlit_app.py`
- `README.md`
