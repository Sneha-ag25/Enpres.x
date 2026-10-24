# Enpres.x

## Project objective

Enpres.x is a beginner-friendly hackathon project for CL-03. Its objective is to predict building electricity consumption using historical building energy data and related features.

## Planned technology

- Python
- pandas
- scikit-learn
- Streamlit

## Next steps

1. Obtain and inspect the building energy consumption dataset.
2. Prepare and clean the raw data.
3. Explore the data and identify useful features.
4. Train and evaluate a machine-learning model.
5. Save the trained model and document the results.
6. Build a Streamlit dashboard for predictions and visualizations.
7. Test the complete project and prepare the hackathon presentation.

## Run the dashboard

From the project folder, open PowerShell and run:

```powershell
cd C:\Users\sneha\Desktop\Enpres.x
py -m streamlit run app\streamlit_app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

The dashboard uses the saved baseline model outputs and held-out test predictions. It does not retrain the model. Missing electricity readings are shown as data-quality gaps, not as zero consumption.

The dashboard also displays an empirical 80% prediction interval calibrated from validation-period residuals. Actual test coverage may differ from the nominal 80% target.

## Optional verified 24-hour forecast input

The dashboard does not invent future weather. A live 24-hour forecast is shown only when `data/processed/future_weather.csv` exists with exactly 24 hourly rows and these columns:

```text
timestamp,airTemperature,cloudCoverage,dewTemperature,precipDepth1HR,precipDepth6HR,seaLvlPressure,windDirection,windSpeed
```

The file must contain complete weather values for the 24 hours immediately after the latest historical timestamp. The forecast uses the saved model recursively: earlier predictions become later lag inputs, and future actual electricity values are never used.

## Phase 2 extensions

### Meter data-quality flags

The dashboard flags, but does not modify, electricity readings:

- **Missing reading:** `electricity` is absent or non-numeric.
- **Long missing gap:** at least 24 consecutive missing hourly readings for the same building.
- **Suspicious spike:** an observed reading is above 3× its centered 24-hour rolling median and more than 6 rolling median absolute deviations above that baseline. This is a transparent screening rule, not a confirmed meter fault.
- **Confirmed faulty reading:** none are asserted from the current files because no maintenance or meter-validation labels are available.

Raw data remains unchanged, and missing targets are never filled with zero.

### Potential audit candidates

The dashboard ranks only the five saved baseline buildings using held-out predictions. Residual is defined as:

```text
observed electricity - saved model prediction
```

A building must have at least 168 observed test hours and a positive-residual streak of at least 6 hours. The screening score is:

```text
sustained positive-residual hours × mean positive residual
```

This is a prioritization aid for investigation, not evidence of energy waste. Weather mismatch, occupancy, model error, and data quality can all contribute to a positive residual.

### Illustrative retrofit scenarios

The selected building view shows separate hypothetical 10% and 8% reduction scenarios when observed held-out electricity is available. For each scenario:

```text
estimated reduction = observed baseline energy × assumed reduction percentage
```

These are illustrative energy estimates only. They are not measured or guaranteed savings, and no financial savings are assumed or double-counted.

### Unseen-site evaluation

The fixed Panther model was evaluated on the separate `Gator` site without refitting on Gator electricity targets. The evaluation is documented in `reports/unseen_site_evaluation.md` and can be reproduced with:

```powershell
py src\evaluate_unseen_site.py
```

The comparison keeps only rows with observed electricity, valid past-only lag features, and complete values for all eight weather inputs, then compares the model and same-hour-previous-day benchmark on exactly the same rows. Gator's `yearbuilt` and `eui` metadata are missing for all evaluated buildings, so the saved training-fitted numeric imputer is used; this is an important domain-shift limitation. The result is one site-specific check and does not prove universal generalization.
