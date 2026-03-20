import json
import hashlib

def calculate_severity(current_data, last_memory_state=None):
    """
    Evaluates sweep deltas strictly based on mathematically deterministic vectors.
    1. Absolute Severity
    2. Cross-domain correlation
    """
    score = 0
    if "HIGH" in current_data.get('tier_1', ''):
        score += 3
    if "22" in current_data.get('tier_2', ''):
        score += 2
        
    if score >= 5:
        return "FLASH"
    elif score >= 3:
        return "PRIORITY"
    return "ROUTINE"

def cross_correlate(current_data):
    """
    Implementation of XDR-style correlation connecting Geopolitical (Tier 1) 
    with Yields (Tier 2).
    """
    alert_level = calculate_severity(current_data)
    return {
        "raw_signals": current_data,
        "correlation_score": alert_level,
        "hash_id": hashlib.sha256(json.dumps(current_data).encode()).hexdigest()[:8]
    }
