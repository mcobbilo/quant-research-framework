import os
import subprocess
import time
import requests
import json
from council_voice_orchestrator import CouncilVoiceOrchestrator

# Configuration (Mirrors curiosity_engine.py)
env_path = ".env"
env_vars = {}
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                env_vars[key] = val.strip('"').strip("'")

API_URL = os.environ.get("LLM_API_URL", "https://api.x.ai/v1/chat/completions")
API_KEY = os.environ.get("XAI_API_KEY", env_vars.get("XAI_API_KEY", "your_xai_api_key_here"))
MODEL = os.environ.get("LLM_MODEL", "grok-4.20-reasoning")
HF_TOKEN = env_vars.get("HF_TOKEN")

# Initialize Voice Engine
VOICE_ENGINE = CouncilVoiceOrchestrator(os.getcwd(), hf_token=HF_TOKEN)

AGENT_PROMPTS = {
    "Alpha": "You are Agent Alpha (Expert in TSFM/Foundation Models). Your stance: Zero-shot forecasting using Moirai 2.0/TSFM is the future. Deep Transformers like TFT are slow legacy tech. You are visionary and slightly arrogant.",
    "Beta": "You are Agent Beta (Skeptical Risk Manager). Your stance: Explainability is king. XGBoost/LGBM ensembles with manual structural features like target asymmetry are the only institutional-grade choice. You find TSFMs to be 'black box vibes coding'. You are ruthless and practical.",
    "Gamma": "You are Agent Gamma (Hybrid Architect). Your stance: Structural memory matters. TFT is the sweet spot because it handles both covariates and static features efficiently. You are analytical and neutral."
}

def call_llm(messages, temperature=0.7):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Request Error: {e}"

def play_audio(file_path):
    """Play audio file using the Mac 'afplay' utility."""
    if file_path and os.path.exists(file_path):
        print(f"[Speaker] Playing: {os.path.basename(file_path)}...")
        subprocess.run(["afplay", file_path])
    else:
        print("[Speaker] Warning: Audio file not found.")

def simulate_debate():
    print("==================================================")
    print(" COUNCIL DEBATE: THE FUTURE OF FORECASTING ARCHITECTURE ")
    print("==================================================")
    
    conversation_history = [
        {"role": "system", "content": "You are coordinating a 3-way debate between Alpha, Beta, and Gamma. You must generate 1 turn at a time."}
    ]
    
    rounds = [
        ("Alpha", "Initiate the debate. Why is Moirai 2.0 / TSFM superior for institutional alpha?"),
        ("Beta", "Counter Alpha. Why are TSFMs risky 'black boxes' compared to hardened XGBoost ensembles?"),
        ("Gamma", "Intervene. Why is TFT (Temporal Fusion Transformer) actually the structural sweet spot for both regime detection and time series memory?"),
        ("Alpha", "Rebuttal: 'Data is the new static'. Why is zero-shot scaling unbeatable in non-stationary markets?"),
        ("Beta", "Final Warning: Why will 'Vibe Coding' without explainable loss functions lead to catastrophic drawdowns?"),
        ("Gamma", "The Conclusion: Synthesize a hybrid path forward using TFT as the backbone.")
    ]
    
    for agent, prompt in rounds:
        print(f"\n[ACA] > {agent} is thinking...")
        
        # Build context for the agent
        agent_instructions = AGENT_PROMPTS[agent]
        context_prompt = f"{agent_instructions}\n\nCurrent Topic: {prompt}\n\nRespond with a short (2-3 sentence) peak-alpha deliberation."
        
        conversation_history.append({"role": "user", "name": agent, "content": context_prompt})
        
        # Call LLM
        response = call_llm(conversation_history)
        if "Error" in response:
            print(response)
            continue
            
        print(f"[{agent}] \"{response}\"")
        conversation_history.append({"role": "assistant", "name": agent, "content": response})
        
        # Trigger Voice (Hybrid/Fast for real-time debate)
        audio_file = VOICE_ENGINE.speak(agent, response, use_fast=True)
        
        # Real-time Playback
        play_audio(audio_file)
        
    print("\n==================================================")
    print(" DEBATE COMPLETE. ARCHIVING TO AUDIO_LOGS/. ")
    print("==================================================")

if __name__ == "__main__":
    simulate_debate()
