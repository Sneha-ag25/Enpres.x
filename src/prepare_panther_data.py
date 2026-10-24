"""Prepare a long-format Panther electricity dataset."""

from collections import Counter
from pathlib import Path
import csv


WORKSPACE = Path(__file__).resolve().parents[1]
RAW = WORKSPACE / "data" / "raw"
PROCESSED = WORKSPACE / "data" / "processed"
REPORTS = WORKSPACE / "reports"
OUTPUT = PROCESSED / "panther_electricity.csv"
SUMMARY = REPORTS / "panther_data_summary.md"

METADATA_FIELDS = [
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "sqft",
    "lat",
    "lng",
    "timezone",
    "yearbuilt",
    "numberoffloors",
    "occupants",
    "energystarscore",
    "eui",
    "site_eui",
    "source_eui",
    "leed_level",
    "rating",
]

WEATHER_FIELDS = [
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "precipDepth1HR",
    "precipDepth6HR",
    "seaLvlPressure",
    "windDirection",
    "windSpeed",
]

OUTPUT_FIELDS = [
    "timestamp",
    "building_id",
    "site_id",
    "electricity",
    *METADATA_FIELDS,
    *WEATHER_FIELDS,
]


def read_panther_metadata():
    metadata = {}
    with (RAW / "metadata.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["site_id"] == "Panther":
                metadata[row["building_id"]] = row
    return metadata


def read_panther_weather():
    weather = {}
    counts = Counter()
    with (RAW / "weather.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["site_id"] != "Panther":
                continue
            timestamp = row["timestamp"]
            counts[timestamp] += 1
            weather[timestamp] = [row[field] for field in WEATHER_FIELDS]
    return weather, counts


def get_electricity_buildings():
    with (RAW / "electricity_cleaned.csv").open(encoding="utf-8", newline="") as source:
        header = next(csv.reader(source))
    return header, [name for name in header[1:] if name.startswith("Panther_")]


def prepare(metadata, weather, electricity_header, electricity_buildings):
    matched = [name for name in electricity_buildings if name in metadata]
    unmatched = [name for name in electricity_buildings if name not in metadata]
    metadata_only = sorted(set(metadata) - set(matched))
    indices = [electricity_header.index(name) for name in matched]

    missing = {field: 0 for field in OUTPUT_FIELDS}
    timestamp_counts = Counter()
    output_rows = 0
    first_timestamp = None
    last_timestamp = None

    with (
        (RAW / "electricity_cleaned.csv").open(encoding="utf-8", newline="") as source,
        OUTPUT.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.reader(source)
        next(reader)
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(OUTPUT_FIELDS)

        for row in reader:
            if not row:
                continue
            timestamp = row[0]
            timestamp_counts[timestamp] += 1
            first_timestamp = first_timestamp or timestamp
            last_timestamp = timestamp
            weather_values = weather.get(timestamp, [""] * len(WEATHER_FIELDS))

            for building_id, index in zip(matched, indices):
                metadata_row = metadata[building_id]
                values = [timestamp, building_id, "Panther", row[index]]
                values.extend(metadata_row[field] for field in METADATA_FIELDS)
                values.extend(weather_values)
                writer.writerow(values)
                output_rows += 1
                for field, value in zip(OUTPUT_FIELDS, values):
                    if value == "":
                        missing[field] += 1

    return {
        "matched": matched,
        "unmatched": unmatched,
        "metadata_only": metadata_only,
        "missing": missing,
        "timestamp_counts": timestamp_counts,
        "output_rows": output_rows,
        "source_rows": sum(timestamp_counts.values()),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def validate_output(expected_rows):
    with OUTPUT.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        rows = 0
        first_row = None
        last_row = None
        for row in reader:
            first_row = first_row or row
            last_row = row
            rows += 1
    if header != OUTPUT_FIELDS:
        raise RuntimeError("Prepared output header does not match the expected schema")
    if rows != expected_rows:
        raise RuntimeError(f"Prepared output row count mismatch: {rows} != {expected_rows}")
    return header, first_row, last_row


def write_summary(metadata, weather, weather_counts, electricity_header, electricity_buildings, result, validation):
    weather_duplicate_timestamps = sum(count - 1 for count in weather_counts.values() if count > 1)
    electricity_duplicate_timestamps = sum(
        count - 1 for count in result["timestamp_counts"].values() if count > 1
    )
    missing_weather_timestamps = len(
        set(result["timestamp_counts"]) - set(weather)
    )
    header, first_row, last_row = validation
    output_rows = result["output_rows"]

    lines = [
        "# Panther Data Preparation Summary",
        "",
        f"Workspace: `{WORKSPACE}`",
        "",
        "## Source files inspected",
        "",
        "- `data/raw/electricity_cleaned.csv`",
        "- `data/raw/metadata.csv`",
        "- `data/raw/weather.csv`",
        "",
        f"- Electricity columns: **{len(electricity_header)}** (`timestamp` plus building IDs)",
        "- Metadata columns: **32**",
        "- Weather columns: **10**",
        "",
        "## Panther matching",
        "",
        f"- Panther electricity building columns: **{len(electricity_buildings)}**",
        f"- Panther metadata building records: **{len(metadata)}**",
        f"- Matched building IDs used: **{len(result['matched'])}**",
        f"- Electricity IDs without metadata: **{len(result['unmatched'])}**",
        f"- Metadata IDs without electricity columns: **{len(result['metadata_only'])}**",
        f"- Panther weather rows: **{sum(weather_counts.values()):,}**",
        "",
        "## Timestamp checks",
        "",
        f"- Electricity rows read: **{result['source_rows']:,}**",
        f"- Electricity coverage: `{result['first_timestamp']}` through `{result['last_timestamp']}`",
        f"- Panther weather coverage: `{min(weather)}` through `{max(weather)}`",
        f"- Duplicate electricity timestamps: **{electricity_duplicate_timestamps}**",
        f"- Duplicate Panther weather timestamps: **{weather_duplicate_timestamps}**",
        f"- Electricity timestamps without matching Panther weather: **{missing_weather_timestamps}**",
        "",
        "The prepared file is long format: each row represents one `(timestamp, building_id)` pair.",
        "Repeated timestamps are therefore expected in the prepared output.",
        "",
        "## Prepared output",
        "",
        "- File: `data/processed/panther_electricity.csv`",
        f"- Rows: **{output_rows:,}**",
        f"- Columns: **{len(header)}**",
        "- Columns:",
        "",
        "```text",
        ", ".join(header),
        "```",
        "",
        "## Missing values",
        "",
        f"Counts are across all **{output_rows:,}** prepared rows.",
        "",
    ]
    for field in OUTPUT_FIELDS:
        count = result["missing"][field]
        percentage = count / output_rows * 100 if output_rows else 0
        lines.append(f"- `{field}`: {count:,} ({percentage:.2f}%)")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Original CSV files in `data/raw/` were not modified.",
            "- Only buildings with both Panther electricity and metadata records were included.",
            "- No model training or dashboard development was performed.",
            "",
        ]
    )
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print(f"OUTPUT={OUTPUT}")
    print(f"SUMMARY={SUMMARY}")
    print(f"OUTPUT_ROWS={output_rows}")
    print(f"OUTPUT_COLUMNS={len(header)}")
    print(f"MATCHED_BUILDINGS={len(result['matched'])}")
    print(f"ELECTRICITY_DUPLICATE_TIMESTAMPS={electricity_duplicate_timestamps}")
    print(f"WEATHER_DUPLICATE_TIMESTAMPS={weather_duplicate_timestamps}")
    print(f"MISSING_WEATHER_TIMESTAMPS={missing_weather_timestamps}")
    print(f"UNMATCHED_ELECTRICITY_BUILDINGS={len(result['unmatched'])}")
    print(f"METADATA_ONLY_BUILDINGS={len(result['metadata_only'])}")
    print("OUTPUT_HEADER=" + ",".join(header))
    print("FIRST_OUTPUT_ROW=" + "|".join(first_row or []))
    print("LAST_OUTPUT_ROW=" + "|".join(last_row or []))
    print("MISSING_COUNTS")
    for field in OUTPUT_FIELDS:
        print(f"{field}={result['missing'][field]}")


def main():
    if SUMMARY.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {SUMMARY}")
    metadata = read_panther_metadata()
    weather, weather_counts = read_panther_weather()
    electricity_header, electricity_buildings = get_electricity_buildings()
    result = prepare(metadata, weather, electricity_header, electricity_buildings)
    validation = validate_output(result["output_rows"])
    write_summary(
        metadata,
        weather,
        weather_counts,
        electricity_header,
        electricity_buildings,
        result,
        validation,
    )


if __name__ == "__main__":
    main()
