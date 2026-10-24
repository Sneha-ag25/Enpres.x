# Panther Data Preparation Summary

Workspace: `C:\Users\sneha\Desktop\Enpres.x`

## Source files inspected

- `data/raw/electricity_cleaned.csv`
- `data/raw/metadata.csv`
- `data/raw/weather.csv`

- Electricity columns: **1579** (`timestamp` plus building IDs)
- Metadata columns: **32**
- Weather columns: **10**

## Panther matching

- Panther electricity building columns: **105**
- Panther metadata building records: **136**
- Matched building IDs used: **105**
- Electricity IDs without metadata: **0**
- Metadata IDs without electricity columns: **31**
- Panther weather rows: **17,544**

## Timestamp checks

- Electricity rows read: **17,544**
- Electricity coverage: `2016-01-01 00:00:00` through `2017-12-31 23:00:00`
- Panther weather coverage: `2016-01-01 00:00:00` through `2017-12-31 23:00:00`
- Duplicate electricity timestamps: **0**
- Duplicate Panther weather timestamps: **0**
- Electricity timestamps without matching Panther weather: **0**

The prepared file is long format: each row represents one `(timestamp, building_id)` pair.
Repeated timestamps are therefore expected in the prepared output.

## Prepared output

- File: `data/processed/panther_electricity.csv`
- Rows: **1,842,120**
- Columns: **28**
- Columns:

```text
timestamp, building_id, site_id, electricity, primaryspaceusage, sub_primaryspaceusage, sqm, sqft, lat, lng, timezone, yearbuilt, numberoffloors, occupants, energystarscore, eui, site_eui, source_eui, leed_level, rating, airTemperature, cloudCoverage, dewTemperature, precipDepth1HR, precipDepth6HR, seaLvlPressure, windDirection, windSpeed
```

## Missing values

Counts are across all **1,842,120** prepared rows.

- `timestamp`: 0 (0.00%)
- `building_id`: 0 (0.00%)
- `site_id`: 0 (0.00%)
- `electricity`: 364,968 (19.81%)
- `primaryspaceusage`: 0 (0.00%)
- `sub_primaryspaceusage`: 0 (0.00%)
- `sqm`: 0 (0.00%)
- `sqft`: 0 (0.00%)
- `lat`: 0 (0.00%)
- `lng`: 0 (0.00%)
- `timezone`: 0 (0.00%)
- `yearbuilt`: 0 (0.00%)
- `numberoffloors`: 1,842,120 (100.00%)
- `occupants`: 1,842,120 (100.00%)
- `energystarscore`: 1,842,120 (100.00%)
- `eui`: 0 (0.00%)
- `site_eui`: 1,842,120 (100.00%)
- `source_eui`: 1,842,120 (100.00%)
- `leed_level`: 0 (0.00%)
- `rating`: 1,842,120 (100.00%)
- `airTemperature`: 315 (0.02%)
- `cloudCoverage`: 791,910 (42.99%)
- `dewTemperature`: 315 (0.02%)
- `precipDepth1HR`: 945 (0.05%)
- `precipDepth6HR`: 1,774,290 (96.32%)
- `seaLvlPressure`: 33,915 (1.84%)
- `windDirection`: 52,395 (2.84%)
- `windSpeed`: 0 (0.00%)

## Notes

- Original CSV files in `data/raw/` were not modified.
- Only buildings with both Panther electricity and metadata records were included.
- No model training or dashboard development was performed.
