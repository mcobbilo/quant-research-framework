import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.agents.option_dealer import option_dealer_node

def test_infused_node():
    print("Testing HST-Jamba Infused Option Dealer Node...")
    
    mock_state = {
        "spy_data": "SPY: 512.30 (-0.05%), SMA_50 crossover confirmed. Delta skew is flattening.",
        "vix_data": "VIX: 14.50 (+2.1%), Mean reversion cluster active. IV rank: 12%."
    }
    
    # This should trigger Jamba initialization on MPS and Nemotron-4 call
    result = option_dealer_node(mock_state)
    
    print("\n--- INFUSION RESULT ---")
    print(f"Direction: {result.get('option_dealer_direction')}")
    print(f"Rationale: {result.get('option_dealer_rationale')[:300]}...")
    
    if "Jamba" in result.get('option_dealer_rationale', "") or "Regime" in result.get('option_dealer_rationale', ""):
         print("\n[SUCCESS] Jamba-Ensemble alignment confirmed in reasoning.")
    else:
         print("\n[WARNING] Jamba signal not explicitly cited, but node executed.")

if __name__ == "__main__":
    test_infused_node()
