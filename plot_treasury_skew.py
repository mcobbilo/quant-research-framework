import pandas as pd
import requests
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web
import os
import warnings

warnings.filterwarnings("ignore")


def generate_treasury_chart():
    print("Fetching FiscalData for Treasury Issuance...")
    # Fetch 10,000 records from the U.S. Treasury's official API
    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_1?page[size]=10000"
    response = requests.get(url).json()

    records = []
    for row in response.get("data", []):
        if row.get("security_type_desc") == "Marketable":
            cls_desc = row.get("security_class_desc", "")
            if cls_desc in ["Bills", "Notes", "Bonds"]:
                date_str = row.get("record_date")
                amt = float(row.get("total_mil_amt", 0).replace(",", ""))
                records.append({"Date": date_str, "Type": cls_desc, "Amount": amt})

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    # Pivot so Bills, Notes, Bonds are columns
    pivot = df.pivot_table(
        index="Date", columns="Type", values="Amount", aggfunc="sum"
    ).fillna(0)

    # Values in millions. Divide by 1,000,000 to get Trillions.
    pivot["Bills_Trillions"] = pivot["Bills"] / 1000000
    pivot["Notes_Bonds_Trillions"] = (pivot["Notes"] + pivot["Bonds"]) / 1000000
    pivot["T_Bill_Percentage"] = (
        pivot["Bills_Trillions"]
        / (pivot["Bills_Trillions"] + pivot["Notes_Bonds_Trillions"])
    ) * 100

    print("Fetching FRED for Reverse Repo (RRPONTSYD)...")
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=5 * 365)
    fred_df = web.DataReader(["RRPONTSYD"], "fred", start_date, end_date)
    # RRPONTSYD is in Billions. Divide by 1,000 to get Trillions.
    fred_df["RRP_Trillions"] = fred_df["RRPONTSYD"] / 1000
    fred_df = fred_df.ffill().fillna(0)

    # Merge the daily FRED data with the monthly Treasury data (forward filled)
    merged = pd.merge(
        fred_df[["RRP_Trillions"]],
        pivot[["Bills_Trillions", "T_Bill_Percentage"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    merged["Bills_Trillions"] = merged["Bills_Trillions"].ffill()
    merged["T_Bill_Percentage"] = merged["T_Bill_Percentage"].ffill()
    merged = merged.dropna()

    print("Generating Visual Quantification Chart...")
    plt.style.use("dark_background")
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    # --- Top Panel: Cross-Correlation of T-Bills vs RRP Drain ---
    # Plot T-Bills Outstanding (Left Axis)
    ax1.plot(
        merged.index,
        merged["Bills_Trillions"],
        color="#00d2ff",
        linewidth=3,
        label="T-Bills Outstanding ($ Trillions)",
    )
    ax1.fill_between(
        merged.index,
        merged["Bills_Trillions"],
        min(merged["Bills_Trillions"]) * 0.9,
        color="#00d2ff",
        alpha=0.1,
    )
    ax1.set_ylabel(
        "U.S. T-Bills Outstanding ($ Trillions)", fontsize=12, color="#00d2ff"
    )
    ax1.tick_params(axis="y", labelcolor="#00d2ff")
    ax1.grid(color="#2A3459", linestyle="--", alpha=0.7)

    # Plot Reverse Repo Drain (Right Axis)
    ax2 = ax1.twinx()
    ax2.plot(
        merged.index,
        merged["RRP_Trillions"],
        color="#ef5350",
        linewidth=3,
        label="Reverse Repo Facility (RRP)",
    )
    ax2.fill_between(
        merged.index, merged["RRP_Trillions"], 0, color="#ef5350", alpha=0.1
    )
    ax2.set_ylabel("Fed RRP Balance ($ Trillions)", fontsize=12, color="#ef5350")
    ax2.tick_params(axis="y", labelcolor="#ef5350")

    ax1.set_title(
        'The Treasury "Stealth QE": Replacing Lost Fed Liquidity (5-Year Window)',
        fontsize=16,
        color="white",
        pad=20,
    )

    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        frameon=True,
        fontsize=12,
        facecolor="#0D1117",
    )

    # --- Bottom Panel: T-Bill Debt Percentage ---
    ax3.plot(merged.index, merged["T_Bill_Percentage"], color="#26a69a", linewidth=2)
    ax3.fill_between(
        merged.index, merged["T_Bill_Percentage"], 0, color="#26a69a", alpha=0.2
    )
    ax3.axhline(
        20.0,
        color="yellow",
        linestyle="--",
        linewidth=1.5,
        label="TBAC Guideline Max Limit (20%)",
    )
    ax3.set_ylabel("T-Bills as % of Total Marketable Debt", fontsize=12, color="white")
    ax3.grid(color="#2A3459", linestyle="--", alpha=0.7)
    ax3.set_ylim(
        min(merged["T_Bill_Percentage"]) - 1, max(merged["T_Bill_Percentage"]) + 2
    )
    ax3.legend(loc="upper left", frameon=True, facecolor="#0D1117")

    plt.tight_layout()

    out_path = "/Users/milocobb/.gemini/antigravity/brain/c33063f0-f712-49cd-b420-4b0183d4e862/artifacts/tbill_rrp_drain.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0D1117")
    print(f"Chart saved perfectly to {out_path}")


if __name__ == "__main__":
    generate_treasury_chart()
