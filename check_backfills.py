import pandas as pd
import os

df = pd.read_csv(
    "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/market_data_inspection.csv"
)
report_lines = []

for col in df.columns:
    if col == "Date":
        continue

    first_val = df.loc[0, col]

    # To handle floating point inaccuracies in constant backfills
    if pd.api.types.is_numeric_dtype(df[col]):
        diff_mask = abs(df[col] - first_val) > 1e-6
    else:
        diff_mask = df[col] != first_val

    if diff_mask.any():
        idx = diff_mask.idxmax()
        # idx is the integer row index where the value first changes away from the initial backfilled value
        start_date = df.loc[idx, "Date"]

        # We define a "bogus backfill" as anything that remains perfectly constant for more than 40 trading days (~2 months)
        # Macro data changes at least once a month (approx 20 trading days).
        # Anything holding the exact same floating point value for 40+ days is 99% guaranteed to be a backward fill.

        report_lines.append(
            {"Feature": col, "Constant_Days": idx, "True_Start_Date": start_date}
        )
    else:
        # It never changes?
        report_lines.append(
            {
                "Feature": col,
                "Constant_Days": len(df),
                "True_Start_Date": "NEVER_CHANGES",
            }
        )

# Sort by the number of constant days in descending order to highlight the worst offenders first
report_lines.sort(key=lambda x: x["Constant_Days"], reverse=True)

# Generate Markdown artifact
artifact_path = os.path.expanduser(
    "~/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/backfill_analysis.md"
)
with open(artifact_path, "w") as f:
    f.write("# Data Integrity: Backfill Analysis\n")
    f.write(
        "This table shows the true start dates for all features in the dataset. Variables with hundreds or thousands of 'Constant Days' at the beginning were heavily backward-filled with bogus data due to `pandas.bfill()`.\n\n"
    )
    f.write("| Feature | True Start Date | Backfilled Days (Approx) |\n")
    f.write("|---------|-----------------|--------------------------|\n")
    for row in report_lines:
        if row["Constant_Days"] > 20:  # Only show significant backfills
            f.write(
                f"| **{row['Feature']}** | {row['True_Start_Date']} | {row['Constant_Days']} Trading Days |\n"
            )

print("Artifact written.")
