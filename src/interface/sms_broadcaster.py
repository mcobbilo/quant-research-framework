import subprocess
import os
import requests
from datetime import datetime

def send_text_alert(target_allocation, spy_price, crash_prob, shap_drivers, phone_number="8172879263"):
    """
    Hooks into macOS kernel via osascript to natively dispatch SMS or iMessage.
    
    target_allocation: string e.g. "100% Cash" or "50% SPY / 50% Cash"
    crash_prob: float (e.g. 0.85 for 85%)
    shap_drivers: list of top 3 features (e.g. ["CVR3_BUY_SIGNAL", ...])
    """
    
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if crash_prob >= 0.80:
        alert_type = "🚨 CRITICAL PANIC 🚨"
    elif crash_prob > 0.45:
        alert_type = "⚠️ ELEVATED RISK ⚠️"
    else:
        alert_type = "🟢 REGIME STABLE 🟢"
        
    text_body = f"{alert_type}\n"
    text_body += f"Swarm Update: {date_str}\n"
    text_body += f"SPY Frame: ${spy_price:.2f}\n"
    text_body += f"Systemic Crash Risk: {crash_prob:.1%}\n"
    text_body += f"Target Allocation: {target_allocation}\n\n"
    
    text_body += "Forensic SHAP Rationale:\n"
    for i, driver in enumerate(shap_drivers, 1):
        text_body += f" {i}. {driver}\n"
        
    print(f"[SMS Engine] Compiling AppleScript payload to target cell number {phone_number}...")
    
    applescript = '''
    on run {targetNumber, messagePayload}
        tell application "Messages"
            try
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy targetNumber of targetService
                send messagePayload to targetBuddy
            on error
                try
                    set targetService to 1st service whose service type = SMS
                    set targetBuddy to buddy targetNumber of targetService
                    send messagePayload to targetBuddy
                on error
                    set targetBuddy to participant targetNumber of account 1
                    send messagePayload to targetBuddy
                end try
            end try
        end tell
    end run
    '''
    
    try:
        subprocess.run(['osascript', '-e', applescript, phone_number, text_body], check=True, capture_output=True, text=True)
        print(f"[SMS Engine] SUCCESS: Native macOS Text Alert safely dispatched payload to {phone_number}.")
    except subprocess.CalledProcessError as e:
        print(f"[SMS Engine] ERROR: Failed to hijack Messages Application: {e.stderr}")

if __name__ == "__main__":
    print("[SMS Broadcaster] Synthesizing physical diagnostic test message...")
    dummy_allocation = "50% SPY / 50% Cash"
    dummy_spy = 598.42
    dummy_prob = 0.52
    dummy_shap = ["VIX_BB_WIDTH", "CVR3_SELL_SIGNAL", "HT_Credit_Velocity"]
    
    send_text_alert(dummy_allocation, dummy_spy, dummy_prob, dummy_shap)
