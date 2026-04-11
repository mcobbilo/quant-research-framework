import os
import re
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_ADV = os.path.join(BASE_DIR, "_NYADV.csv")
CSV_DEC = os.path.join(BASE_DIR, "_NYdec.csv")
CSV_UPV = os.path.join(BASE_DIR, "_NYupv.csv")
CSV_DNV = os.path.join(BASE_DIR, "_NYdnv.csv")
CSV_UNC = os.path.join(BASE_DIR, "_NYADu.csv")
CSV_HGH = os.path.join(BASE_DIR, "_NYhgh.csv")
CSV_LOW = os.path.join(BASE_DIR, "_NYlow.csv")


def scrape_market_diet():
    print("[Playwright] Launching Chromium to bypass WAF...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Using a distinct user-agent to avoid immediate bot detection
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            print("[Breadth Scraper] Navigating to Wall Street Journal Market Diary...")
            page.goto(
                "https://www.wsj.com/market-data/stocks/us",
                wait_until="domcontentloaded",
                timeout=60000,
            )

            # Wait for dynamic tables to render
            time.sleep(5)

            visible_text = page.locator("body").inner_text()

            # Fallback data if extraction fails
            advances, declines, unchanged = "0", "0", "0"
            up_vol, down_vol = "0", "0"
            highs, lows = "0", "0"

            # Extract basic integers using robust text anchoring
            # WSJ typically displays: "Advancing\t615\t1,047" (first line is issues, second is volume)

            adv_matches = re.findall(r"Advancing[ \t]+([\d,]+)", visible_text)
            dec_matches = re.findall(r"Declining[ \t]+([\d,]+)", visible_text)
            unc_matches = re.findall(r"Unchanged[ \t]+([\d,]+)", visible_text)
            hgh_matches = re.findall(r"New Highs[ \t]+([\d,]+)", visible_text)
            low_matches = re.findall(r"New Lows[ \t]+([\d,]+)", visible_text)

            if len(adv_matches) >= 2:
                advances = adv_matches[0].replace(",", "")
                up_vol = adv_matches[1].replace(",", "")
            if len(dec_matches) >= 2:
                declines = dec_matches[0].replace(",", "")
                down_vol = dec_matches[1].replace(",", "")
            if len(unc_matches) >= 1:
                unchanged = unc_matches[0].replace(",", "")
            if len(hgh_matches) >= 1:
                highs = hgh_matches[0].replace(",", "")
            if len(low_matches) >= 1:
                lows = low_matches[0].replace(",", "")

            if advances == "0" and declines == "0":
                print(
                    "[Breadth Scraper] WARNING: Could not parse exact metrics from WSJ Market Diary plain text."
                )
                print("Dumping snapshot of visible text for debugging...")
                with open("/tmp/wsj_scrape_dump.txt", "w") as f:
                    f.write(visible_text)

            return advances, declines, unchanged, up_vol, down_vol, highs, lows

        except Exception as e:
            print(f"[Breadth Scraper] Exception during DOM extraction: {e}")
            return "0", "0", "0", "0", "0", "0", "0"
        finally:
            browser.close()


def append_to_historical_csv(filepath, value):
    today_str = datetime.now().strftime("%m/%d/%Y")

    # Check if we already appended today to prevent duplicate rows on re-runs
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
            if len(lines) > 0 and lines[-1].split(",")[0].strip() == today_str:
                print(
                    f"Skipping {os.path.basename(filepath)} - Today's date ({today_str}) already exists."
                )
                return
    except FileNotFoundError:
        pass  # File doesn't exist yet, create will handle it

    # Schema spacing format matches _NYA200R constraints
    # Date, Open, High, Low, Close, Volume
    # E.g.: "06/25/1971,    3.4700,    3.4700,    3.4700,    3.4700,             0 "

    val_float = float(value)

    with open(filepath, "a") as f:
        # Pre-format with deterministic precision matching the historical CSV lines
        formatted_row = f"{today_str},{val_float:10.4f},{val_float:10.4f},{val_float:10.4f},{val_float:10.4f},             0 \n"
        f.write(formatted_row)

    print(f"Successfully appended {val_float} to {os.path.basename(filepath)}")


if __name__ == "__main__":
    print("-" * 50)
    print("ZeroClaw Internal: Market Breadth Pipeline Execution")
    print("-" * 50)

    adv, dec, unc, upv, dnv, hgh, low = scrape_market_diet()

    print(f"\n[Extracted Metrics as of {datetime.now().strftime('%Y-%m-%d')}]")
    print(f"Advances:  {adv}")
    print(f"Declines:  {dec}")
    print(f"Unchanged: {unc}")
    print(f"Up Vol:    {upv}")
    print(f"Down Vol:  {dnv}")
    print(f"New Highs: {hgh}")
    print(f"New Lows:  {low}\n")

    if adv != "0" or dec != "0":
        print("Appending to local historical CSV database...")
        append_to_historical_csv(CSV_ADV, adv)
        append_to_historical_csv(CSV_DEC, dec)
        append_to_historical_csv(CSV_UNC, unc)
        append_to_historical_csv(CSV_UPV, upv)
        append_to_historical_csv(CSV_DNV, dnv)
        append_to_historical_csv(CSV_HGH, hgh)
        append_to_historical_csv(CSV_LOW, low)
        print("Market Breadth Sync Complete.")
    else:
        print("Execution bypassed appending rows due to zeroized DOM readout.")
