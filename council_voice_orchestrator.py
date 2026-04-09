import os
import subprocess
import time
import json
import numpy as np
try:
    import sphn
except ImportError:
    print("[CouncilVoice] Warning: 'sphn' module not found. Voice functionality may be limited.")
    sphn = None

class CouncilVoiceOrchestrator:
    def __init__(self, framework_path, hf_token=None):
        self.framework_path = framework_path
        self.personaplex_path = os.path.join(framework_path, "personaplex")
        self.audio_logs_dir = os.path.join(framework_path, "audio_logs")
        
        # Load .env manually for tokens
        env_vars = {}
        env_path = os.path.join(framework_path, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        env_vars[k] = v.strip('"').strip("'")
        
        self.hf_token = hf_token or env_vars.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        
        # Ensure audio logs dir exists
        os.makedirs(self.audio_logs_dir, exist_ok=True)
        
        # Agent to Persona Voice Mappings
        self.agent_voices = {
            "Alpha": "NATM1.pt",   # Energetic Male
            "Beta": "NATM0.pt",    # Ruthless/Low Male
            "Gamma": "NATF1.pt"    # Analytical Female
        }

    def speak_fast(self, agent_name, text):
        """Uses edge-tts (fast cloud-based) for immediate feedback."""
        import sys
        voices = {
            "Alpha": "en-US-ChristopherNeural",
            "Beta": "en-US-EricNeural",
            "Gamma": "en-US-JennyNeural"
        }
        voice = voices.get(agent_name, "en-US-AriaNeural")
        timestamp = int(time.time())
        output_wav = os.path.join(self.audio_logs_dir, f"fast_{agent_name}_{timestamp}.mp3")
        
        # Try to find edge-tts in the same bin/ as current python (venv)
        bin_dir = os.path.dirname(sys.executable)
        edge_tts_bin = os.path.join(bin_dir, "edge-tts")
        if not os.path.exists(edge_tts_bin):
            edge_tts_bin = "edge-tts" # Fallback to path

        print(f"[CouncilVoice] Generating FAST speech for {agent_name}...")
        cmd = [edge_tts_bin, "--voice", voice, "--text", text, "--write-media", output_wav]
        subprocess.run(cmd, capture_output=True)
        return output_wav

    def create_silence_wav(self, duration, output_path, sample_rate=24000):
        """Generates a silent wav file for the Moshi driver."""
        import wave
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setframerate(sample_rate)
            wf.setsampwidth(2) # 16-bit
            silence_data = np.zeros(int(duration * sample_rate), dtype=np.int16).tobytes()
            wf.writeframes(silence_data)

    def speak(self, agent_name, text, duration_estimation_factor=0.15, use_fast=False):
        """
        Triggers PersonaPlex offline inference or fast fallback.
        """
        if use_fast:
            return self.speak_fast(agent_name, text)

        if agent_name not in self.agent_voices:
            print(f"[CouncilVoice] Warning: Unknown agent {agent_name}. Using NATF1.pt as default.")
            voice_pt = "NATF1.pt"
        else:
            voice_pt = self.agent_voices[agent_name]

        # ... rest of the existing code ...

        # 1. Estimate duration needed based on word count
        word_count = len(text.split())
        duration = max(5, int(word_count * duration_estimation_factor) + 2)
        
        # 2. Setup paths
        timestamp = int(time.time())
        input_wav = os.path.join(self.audio_logs_dir, f"drive_{agent_name}_{timestamp}.wav")
        output_wav = os.path.join(self.audio_logs_dir, f"{agent_name}_{timestamp}.wav")
        output_text = os.path.join(self.audio_logs_dir, f"{agent_name}_{timestamp}.json")
        
        # 3. Create driving silence
        self.create_silence_wav(duration, input_wav)
        
        # 4. Prepare the command
        cmd = [
            "python3", "-m", "moshi.offline",
            "--input-wav", input_wav,
            "--output-wav", output_wav,
            "--output-text", output_text,
            "--voice-prompt", voice_pt,
            "--text-prompt", text,
            "--device", "cpu", # Explicitly use CPU for Mac stability
            "--cpu-offload" # Essential for mac memory protection
        ]
        
        env = os.environ.copy()
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token

        print(f"[CouncilVoice] Generating speech for {agent_name} ({duration}s drive)...")
        
        # 5. Run it (Note: This is blocking, but we can call it non-blocking if needed)
        try:
            # We run it from the personaplex/moshi directory to ensure imports stay correct
            result = subprocess.run(
                cmd, 
                cwd=os.path.join(self.personaplex_path, "moshi"), 
                env=env,
                capture_output=False, # Show progress in terminal
                timeout=600 # 10 minute timeout per turn
            )
            if result.returncode == 0:
                print(f"[CouncilVoice] Success: {output_wav}")
                return output_wav
            else:
                print(f"[CouncilVoice] Error: {result.stderr}")
                return None
        except Exception as e:
            print(f"[CouncilVoice] Failed to execute Moshi: {e}")
            return None
        finally:
            # Clean up the silent driver WAV
            if os.path.exists(input_wav):
                os.remove(input_wav)

if __name__ == "__main__":
    # Test script if run directly
    orch = CouncilVoiceOrchestrator(os.getcwd())
    orch.speak("Beta", "This is the Risk Manager. Your hypothesis has significant lookahead bias. Abort immediately.")
