import pandas as pd
import yfinance as yf
import glob
import os
import shutil


def run_repair():
    base_dir = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework"
    backup_dir = os.path.join(base_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(base_dir, "_*.csv"))

    # Get truth dates
    spy = yf.download("SPY", start="1970-01-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)

    # Convert index to a pure DatetimeIndex without timezone safely
    valid_days = pd.DatetimeIndex(spy.index.date)

    print(f"Total valid trading days found: {len(valid_days)}")

    for f in sorted(csv_files):
        fname = os.path.basename(f)
        print(f"\nProcessing {fname}...")

        # Backup
        shutil.copy2(f, os.path.join(backup_dir, fname))

        with open(f, "r") as fp:
            first_line = fp.readline()

        df = pd.read_csv(f, header=1, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]

        if "Date" not in df.columns:
            print(f"Skipping {fname}: No 'Date' column")
            continue

        df["Date"] = pd.to_datetime(df["Date"])

        # Determine the logical boundary
        min_date = df["Date"].min()
        max_date = valid_days.max()

        target_dates = valid_days[(valid_days >= min_date) & (valid_days <= max_date)]

        # Make Date the index carefully removing duplicates if any exist
        df.drop_duplicates(subset=["Date"], inplace=True)
        df.set_index("Date", inplace=True)

        # Reindex over valid trading days to inject NaNs where days are missing
        df = df.reindex(target_dates)

        # Track missing count
        missing_count = df["Close"].isnull().sum()
        if missing_count == 0:
            print(" -> No gaps detected. Skipping rewrite.")
            continue

        print(
            f" -> Found {missing_count} missing days. Interpolating mathematically (Linear)..."
        )

        # Apply linear interpolation
        df.interpolate(method="linear", inplace=True)
        # Handle edges (if start/end are NaN, fallback to bfill/ffill)
        df.bfill(inplace=True)
        df.ffill(inplace=True)

        # Reset index and format date back to original
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        df["Date"] = df["Date"].dt.strftime("%m/%d/%Y")

        output_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[output_cols]

        # Write back identically mapped
        with open(f, "w") as fp:
            fp.write(first_line)
            # Add spaces for column alignment visually
            fp.write(
                "      Date,       Open,       High,        Low,      Close,        Volume\n"
            )

            # Formatter for proper padding
            for _, row in df.iterrows():
                fp.write(
                    f"{row['Date']:10}, {row['Open']:10.3f}, {row['High']:10.3f}, {row['Low']:10.3f}, {row['Close']:10.3f}, {int(row['Volume']):12}\n"
                )

    print("\nRepair completed successfully.")


if __name__ == "__main__":
    run_repair()
