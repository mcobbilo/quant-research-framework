import json
import os
import io
import sqlite3
import pandas as pd
import mplfinance as mpf
from google import genai
from google.genai import types
from PIL import Image
import time

# ANSI Color Codes
CYAN = "\033[96m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def execute_vision_pattern_agent(target_date=None):
    print(
        f"{CYAN}[Vision Agent] Connecting to Phase 123 SQLite Infrastructure...{RESET}"
    )

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "market_data.db",
    )
    conn = sqlite3.connect(db_path)

    if target_date:
        query = f"""
        SELECT Date, SPY_OPEN as Open, SPY_HIGH as High, SPY_LOW as Low, SPY_CLOSE as Close, SPY_VOLUME as Volume,
               VIX_OPEN, VIX_HIGH, VIX_LOW, VIX_CLOSE
        FROM core_market_table
        WHERE Date <= '{target_date}'
        ORDER BY Date DESC
        LIMIT 300
        """
    else:
        query = """
        SELECT Date, SPY_OPEN as Open, SPY_HIGH as High, SPY_LOW as Low, SPY_CLOSE as Close, SPY_VOLUME as Volume,
               VIX_OPEN, VIX_HIGH, VIX_LOW, VIX_CLOSE
        FROM core_market_table
        ORDER BY Date DESC
        LIMIT 300
        """
    df = pd.read_sql(query, conn, index_col="Date")
    conn.close()

    # Needs to be sorted chronologically for mplfinance
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Construct the native VIX physics array
    vix_df = pd.DataFrame({
        'Open': df['VIX_OPEN'],
        'High': df['VIX_HIGH'],
        'Low': df['VIX_LOW'],
        'Close': df['VIX_CLOSE'],
        'Volume': 0  # Dummy vector to bypass mplfinance strict checking
    })
    
    # Calculate Bollinger Bands natively on the VIX
    vix_df['SMA_20'] = vix_df['Close'].rolling(window=20).mean()
    vix_df['StdDev'] = vix_df['Close'].rolling(window=20).std()
    vix_df['Upper'] = vix_df['SMA_20'] + (vix_df['StdDev'] * 2)
    vix_df['Lower'] = vix_df['SMA_20'] - (vix_df['StdDev'] * 2)
    
    # Calculate Moving Averages on the broader 300-day window
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Isolate strictly the 90-day focal period for the Vision Agent
    df = df.tail(90)
    vix_df = vix_df.tail(90)

    print(
        f"{CYAN}[Vision Agent] Rendering 90-Day High-Res Candlestick Geometry with 50/200 SMAs & VIX Bollinger Mechanics...{RESET}"
    )

    # Create the chart in memory (BytesIO)
    buf = io.BytesIO()

    # Configure institutional dark mode aesthetics
    mc = mpf.make_marketcolors(
        up="g", down="r", edge="inherit", wick="inherit", volume="in", ohlc="i"
    )
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=False)
    
    # Configure custom subplots for moving averages and VIX panel
    apdict = [
        mpf.make_addplot(df['SMA_50'], color='cyan', width=1.5),
        mpf.make_addplot(df['SMA_200'], color='magenta', width=1.5),
        mpf.make_addplot(vix_df, type='candle', panel=2, ylabel='VIX'),
        mpf.make_addplot(vix_df['Upper'], color='red', panel=2, width=1.0, secondary_y=False),
        mpf.make_addplot(vix_df['Lower'], color='green', panel=2, width=1.0, secondary_y=False),
        mpf.make_addplot(vix_df['SMA_20'], color='yellow', linestyle='dashed', panel=2, width=1.0, secondary_y=False)
    ]

    # [PHASE 145: CONTEXT OPTIMIZATION]
    # Slash image tokens/latency by reducing DPI from 300 to 100 and scale from 1.5 to 1.0.
    # Gemini calculates image tokens by geometric tiling; dropping resolution drops token cost and accelerates inference latency without losing geometric validity.
    mpf.plot(
        df,
        type="candle",
        volume=True,
        style=s,
        addplot=apdict,
        panel_ratios=(4, 1, 2),
        title="SPY 90-Day Institutional Tape (w/ VIX & SMAs)",
        ylabel="Price ($)",
        ylabel_lower="Volume",
        figratio=(16, 9),
        figscale=1.0,
        savefig=dict(fname=buf, dpi=100, facecolor="#121212"),
    )

    buf.seek(0)

    print(
        f"{MAGENTA}[Vision Agent] Initiating Multi-Modal Neural Network Analysis (Gemini 3.1 Pro Preview Vision)...{RESET}"
    )

    from dotenv import load_dotenv

    load_dotenv()

    # Fallback checking if user has API Key injected
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            f"{YELLOW}⚠️ [Warning] GOOGLE_API_KEY not detected. Skipping LLM API call and saving chart to /tmp/spy_vision.png for manual inspection.{RESET}"
        )
        with open("/tmp/spy_vision.png", "wb") as f:
            f.write(buf.read())
        return

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    latest_spy_close = df['Close'].iloc[-1]
    latest_sma_50 = df['SMA_50'].iloc[-1]
    latest_sma_200 = df['SMA_200'].iloc[-1]
    latest_vix_close = vix_df['Close'].iloc[-1]
    latest_vix_upper = vix_df['Upper'].iloc[-1]

    system_prompt = f"""
    You are an elite institutional quantitative trader specializing in structural pattern recognition.
    You will be provided a high-resolution 90-Day SPY candlestick chart.
    
    [CRITICAL NUMERICAL STATE]
    To calculate exact crossovers without visual ambiguity, base your logic on these current mathematically derived values:
    - SPY Latest Close: ${latest_spy_close:.2f}
    - 50-Day SMA (Cyan Line): ${latest_sma_50:.2f}
    - 200-Day SMA (Magenta Line): ${latest_sma_200:.2f}
    - The bottom panel is the VIX (Volatility Index) with its 20-Day Bollinger Bands.
    - VIX Latest Close: {latest_vix_close:.2f}
    - VIX Upper Bollinger Band (Red Line): {latest_vix_upper:.2f}
    
    Your directive:
    1. Scan the geometric price structure (Higher Highs, Lower Lows, Consolidation).
    2. Identify classical technical patterns (Head and Shoulders, Bear Flags, Bull Flags, Falling Wedges, Double Bottom/Top, etc.).
    3. Evaluate the price interaction mathematically using the numeric state provided, noting relationships to the 50-Day and 200-Day SMAs (e.g., trend support/resistance, slope trajectory).
    4. Factor the VIX panel into your analysis: mathematically use the provided VIX values to look for volatility structurally breaking outside the upper Bollinger Band (capitulation).
    5. Output your assessment in strict JSON format matching exactly the schema below.
    
    JSON Schema:
    {{
      "detected_pattern": "short pattern name",
      "trend_classification": "Bullish|Bearish|Neutral",
      "confidence": 0.85,
      "rationale": "2-3 sentences explaining exactly what structural weakness or strength you see, citing the explicit moving averages or VIX dislocations if relevant."
    }}
    """

    try:
        t0 = time.time()
        # Load image stream directly for Gemini
        img = Image.open(buf)

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[system_prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        t1 = time.time()
        eval_time = t1 - t0

        result_json = response.text
        payload = json.loads(result_json)
        payload["inference_latency"] = round(eval_time, 2)

        print("\n" + "=" * 60)
        print(f"{BOLD}{GREEN}👁️  GEMINI VISION PATTERN DIAGNOSTICS{RESET}")
        print("=" * 60)
        print(f"| Pattern Matched:  {payload.get('detected_pattern')}")
        print(
            f"| Trend Vector:     {payload.get('trend_classification')} (Confidence: {payload.get('confidence') * 100:.1f}%)"
        )
        print(
            f"| Execution Speed:  {payload.get('inference_latency')} seconds (Optimized)"
        )
        print("| Logical Rationale:")
        import textwrap

        print(
            textwrap.indent(
                textwrap.fill(payload.get("rationale", ""), width=55), "|   "
            )
        )
        print("=" * 60 + "\n")

        # [TELEGRAM NOTIFICATION INSTRUCTION]
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        if telegram_token and telegram_chat_id:
            import requests

            print(
                f"{CYAN}[Vision Agent] Transmitting visual diagnostic to Executive Telegram Channel...{RESET}"
            )

            caption = "👁️ *GEMINI VISION PATTERN DIAGNOSTICS*\n\n"
            caption += f"📐 *Pattern Matched:* {payload.get('detected_pattern')}\n"
            caption += f"📈 *Trend Vector:* {payload.get('trend_classification')} (Conf: {payload.get('confidence') * 100:.1f}%)\n\n"
            caption += f"🧠 *Rationale:*\n{payload.get('rationale')}"

            url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
            buf.seek(0)

            try:
                r = requests.post(
                    url,
                    data={
                        "chat_id": telegram_chat_id,
                        "caption": caption,
                        "parse_mode": "Markdown",
                    },
                    files={"photo": ("spy_90d.png", buf.getvalue(), "image/png")},
                )
                if r.status_code == 200:
                    print(
                        f"{GREEN}[Vision Agent] Successfully beamed payload to Telegram target.{RESET}"
                    )
                else:
                    print(
                        f"{RED}[Vision Agent] Telegram broadcast failed: {r.text}{RESET}"
                    )
            except Exception as tg_e:
                print(f"{RED}[Vision Agent] Telegram HTTP exception: {tg_e}{RESET}")

        return payload

    except Exception as e:
        print(f"{RED}[Error] Visual Inference failed: {e}{RESET}")


if __name__ == "__main__":
    execute_vision_pattern_agent()
