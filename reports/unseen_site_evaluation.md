# Unseen-Site Evaluation

## Scope and safeguards

The saved Panther-trained model was evaluated without refitting, feature selection, or tuning on the `Gator` site's electricity targets. Raw electricity, metadata, and weather files were read but not modified. The comparison uses the same eligible rows for the model and the same-hour-previous-day benchmark.

The existing training-fitted preprocessing was reused. Missing numeric metadata values are accepted by the saved pipeline's training-fitted median imputer; unknown categorical building IDs are handled by the saved encoder's `handle_unknown=ignore` behavior. No missing electricity targets were filled with zero.

## Site and source availability

- Site selected: **Gator**
- Buildings with electricity columns and metadata: **74**
- Electricity rows: **17,544** across `2016-01-01 00:00:00` through `2017-12-31 23:00:00`
- Missing electricity values: **175,133 (13.49%)**
- Weather rows: **17,544** across `2016-01-01 00:00:00` through `2017-12-31 23:00:00`
- Duplicate weather timestamps: **0**
- Weather rows complete across all eight model weather fields: **635**
- Weather missing-value counts: **airTemperature: 3, cloudCoverage: 7,542, dewTemperature: 3, precipDepth1HR: 9, precipDepth6HR: 16,898, seaLvlPressure: 323, windDirection: 499**
- Metadata missing `yearbuilt`: **74** buildings
- Metadata missing `eui`: **74** buildings
- Metadata missing `sqm`: **0** buildings

Because weather completeness is not universal, the evaluation conservatively retained only rows with all eight weather fields present. The saved preprocessing still handles missing numeric static metadata using its original training-fitted imputer.

## Eligible evaluation rows

- Observed electricity targets: **1,123,123**
- Rows with all required past features: **1,080,189**
- Rows with complete weather: **46,990**
- Final common eligible rows: **39,410** across **73** buildings
- Evaluation period represented: `2016-01-02 07:00:00` through `2017-12-29 13:00:00`

Eligibility required an observed electricity target, valid `lag_1`, `lag_24`, and `rolling_24_mean`, plus complete values for all eight weather fields. The benchmark uses the observed `lag_24` value on exactly these rows.

## Results

| Method | MAE | RMSE | Rows |
|---|---:|---:|---:|
| Fixed Panther model | 43.1093 | 198.3712 | 39,410 |
| Same-hour previous-day benchmark | 13.3591 | 66.5987 | 39,410 |

Model minus benchmark: **MAE +29.7502**, **RMSE +131.7725**. Negative values would indicate lower error for the model; positive values indicate the benchmark was better on this unseen-site slice.

## Interpretation and limitations

This is one cross-site evaluation of the fixed Panther model on Gator. It does not prove universal generalization to other sites. The unseen site's static `yearbuilt` and `eui` metadata are missing for all evaluated buildings and therefore rely on the saved Panther training medians; this is a material domain-shift limitation. Weather completeness also reduced the evaluation sample. The results should be treated as evidence about this specific site and eligible period only.

The dashboard remains Panther-focused; this report does not add the unseen site to the dashboard or alter the saved model.
