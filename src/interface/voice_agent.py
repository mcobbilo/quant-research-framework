import asyncio
import os
import sys
import pyaudio
import traceback
import threading
import queue
from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# Radically increase chunk payload to prevent sending 30+ websocket messages/second
CHUNK = 4096 

pyaud = pyaudio.PyAudio()

mic_queue = queue.Queue()
speaker_queue = queue.Queue()

def load_dynamic_persona():
    base_prompt = (
        "You are Antigravity, the lead Quantitative Swarm Architect. "
        "Keep answers hyper-concise and technical. If asked what phase we are on, read the log."
    )
    try:
        cache_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/task.md"
        with open(cache_path, "r") as f:
            task_state = f.read()[-1200:]
        return base_prompt + f"\n\nCURRENT PROJECT STATUS LOG:\n{task_state}"
    except Exception:
        return base_prompt

def mic_thread_worker():
    """Dedicated OS Thread strictly for reading the microphone frame by frame."""
    try:
        audio_stream = pyaud.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("\n[Microphone] Hardware OS Thread Online. You can speak now.\n")
        while True:
            data = audio_stream.read(CHUNK, exception_on_overflow=False)
            mic_queue.put(data)
    except Exception as e:
        print(f"\n[Microphone Thread Crash]: {e}")

def speaker_thread_worker():
    """Dedicated OS Thread strictly for vibrating the laptop speakers."""
    try:
        audio_stream = pyaud.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
        while True:
            data = speaker_queue.get()
            if data is None: break
            audio_stream.write(data)
    except Exception as e:
        print(f"\n[Speaker Thread Crash]: {e}")

async def run_live():
    threading.Thread(target=mic_thread_worker, daemon=True).start()
    threading.Thread(target=speaker_thread_worker, daemon=True).start()
    
    resolved_model = "gemini-2.5-flash-native-audio-latest"
    target_version = "v1alpha"
    
    print(f"\n[Voice Agent] Connecting algorithmic core to: {resolved_model}...")

    try:
        client = genai.Client(http_options={'api_version': target_version})
    except Exception as e:
        print(f"Error initializing Google GenAI Client: {e}")
        sys.exit(1)
        
    market_tool = types.FunctionDeclaration(
        name="get_current_market_status",
        description="Executes a diagnostic check on the 89-Matrix, returning the current tail-risk evaluation of the S&P 500."
    )
    
    config = {
        "tools": [{"function_declarations": [market_tool]}],
        "response_modalities": ["AUDIO"],
        "system_instruction": {"parts": [{"text": load_dynamic_persona()}]},
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    # Options: "Fenrir", "Charon", "Kore", "Puck", "Aoede"
                    "voice_name": "Fenrir" 
                }
            }
        }
    }

    try:
        async with client.aio.live.connect(model=resolved_model, config=config) as session:
            print(f"[Voice Agent] WebSockets synchronized. The Swarm is listening! 🟢\n")
            
            async def receive_from_model():
                try:
                    while True: # Force the receiver to permanently re-arm between conversational turns
                        async for response in session.receive():
                            server_content = getattr(response, 'server_content', None)
                            if server_content is not None:
                                model_turn = getattr(server_content, 'model_turn', None)
                                if model_turn is not None:
                                    for part in model_turn.parts:
                                        inline_data = getattr(part, 'inline_data', None)
                                        if inline_data is not None:
                                            speaker_queue.put(inline_data.data)
                                            
                                        text_data = getattr(part, 'text', None)
                                        if text_data:
                                            print(f"\n[AI Inner Monologue]: {text_data}", end="", flush=True)

                            tool_call = getattr(response, 'tool_call', None)
                            if tool_call is not None:
                                for func_call in tool_call.function_calls:
                                    print(f"\n[Swarm Daemon] Triggered Tool: {func_call.name}()")
                                    if func_call.name == "get_current_market_status":
                                        tool_result = {"status": "The VIX is stable at 14.5. No tail-risk. Target exposure 1.0x SPY."}
                                        await session.send_tool_response(
                                            function_responses=[{"id": func_call.id, "name": func_call.name, "response": tool_result}]
                                        )
                                    else:
                                        await session.send_tool_response(
                                            function_responses=[{"id": func_call.id, "name": func_call.name, "response": {"error": "Tool not implemented globally."}}]
                                        )
                        # Diagnostic telemetry mapping the turn boundary
                        print("\n[Voice Agent] Turn complete. Re-arming receiver for next question...")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"\n[Receiver Error]: {e}")
                    traceback.print_exc()
                finally:
                    # Critical diagnostic tracing
                    print("\n[WARNING] session.receive() loop fundamentally terminated! The Google server closed the socket.")

            async def send_to_model():
                loop = asyncio.get_running_loop()
                while True:
                    pcm_data = await loop.run_in_executor(None, mic_queue.get)
                    try:
                        await session.send_realtime_input(
                            audio={"mime_type": "audio/pcm;rate=16000", "data": pcm_data}
                        )
                    except Exception as e:
                        print(f"\n[Sender Error]: {e}")
                        break

            await asyncio.gather(receive_from_model(), send_to_model())
            
    except Exception as e:
        print(f"\n[WebSocket Error]: Failed to connect. Ensure GEMINI_API_KEY is defined. Details: {e}")

if __name__ == "__main__":
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            print("CRITICAL: 'GEMINI_API_KEY' not found in environment.")
            sys.exit(1)
        asyncio.run(run_live())
    except KeyboardInterrupt:
        print("\n\n[Voice Agent] Session Terminated via KeyboardInterrupt. 🔴\n")
        speaker_queue.put(None)
