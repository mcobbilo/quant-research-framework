let webSocket = null;
let audioContext = null;
let microphone = null;
let processor = null;

const connectBtn = document.getElementById('connect-btn');
const disconnectBtn = document.getElementById('disconnect-btn');
const statusDot = document.getElementById('socket-status-dot');
const statusText = document.getElementById('socket-status-text');
const consoleBox = document.getElementById('ai-transcript');
const canvas = document.getElementById('visualizer');
const ctx = canvas.getContext('2d');
const matrixStatus = document.getElementById('matrix-status');

let audioBufferQueue = [];
let isPlaying = false;

// Auto-resize canvas
function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function log(msg, type="system") {
    const p = document.createElement('div');
    if(type === "system") {
        p.innerHTML = `<span class="log-prefix">> SYSTEM:</span> ${msg}`;
    } else {
        p.innerHTML = `<span class="log-prefix">> AI SWARM:</span> <span class="log-ai">${msg}</span>`;
    }
    consoleBox.appendChild(p);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

// Float32Array to Base64 16-bit PCM Converter
function encodePcm16(float32Array) {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32Array.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    
    // Convert ArrayBuffer to Base64
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

// Audio queue engine to accurately render raw PCM arrays
async function processAudioQueue() {
    if (isPlaying || audioBufferQueue.length === 0) return;
    isPlaying = true;

    const base64Audio = audioBufferQueue.shift();
    const binary = atob(base64Audio);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }

    try {
        // Convert raw Int16 PCM to Float32 for Web Audio Graph
        const int16Array = new Int16Array(bytes.buffer);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }

        // Gemini strictly outputs audio at 24kHz natively
        const audioBuffer = audioContext.createBuffer(1, float32Array.length, 24000);
        audioBuffer.getChannelData(0).set(float32Array);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.onended = () => {
            isPlaying = false;
            processAudioQueue(); // Process next in queue
        };
        source.start(0);
        drawWaveform(); // Trigger visualizer while playing
    } catch (e) {
        log(`System Audio Decoder Failure: ${e.message}`, "system");
        isPlaying = false;
        processAudioQueue();
    }
}

// Sine wave animation
function drawWaveform() {
    if (!isPlaying) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.strokeStyle = '#00f0ff';
        ctx.lineWidth = 2;
        ctx.stroke();
        return;
    }

    requestAnimationFrame(drawWaveform);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    
    // Dynamic waveform generator
    const time = Date.now() / 1000;
    for (let i = 0; i < canvas.width; i++) {
        const amplitude = Math.sin(time * 5 + i * 0.05) * 40 * Math.sin(time * 2);
        ctx.lineTo(i, canvas.height / 2 + amplitude);
    }
    
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 3;
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#00f0ff';
    ctx.stroke();
    ctx.shadowBlur = 0;
}
drawWaveform();

async function connectToSwarm() {
    log("Fetching System Credentials from Sandbox Controller...");
    const credResp = await fetch('/api/credentials');
    const creds = await credResp.json();
    
    const sysResp = await fetch('/api/sys-instruction');
    const sys = await sysResp.json();

    const host = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent";
    const url = `${host}?key=${creds.key}`;

    webSocket = new WebSocket(url);

    webSocket.onopen = async () => {
        statusDot.className = "dot connected";
        statusText.innerText = "ONLINE";
        connectBtn.classList.add("hidden");
        disconnectBtn.classList.remove("hidden");
        log("WebSocket connected. Exchanging Initial Configuration...");

        // Send initialization config
        const initialConfig = {
            setup: {
                model: "models/gemini-2.5-flash-native-audio-latest",
                systemInstruction: { parts: [{ text: sys.instruction }] },
                tools: [{ functionDeclarations: [{
                    name: "get_current_market_status",
                    description: "Executes a diagnostic check on the 89-Matrix, returning the current tail-risk evaluation of the S&P 500."
                }]}],
                generationConfig: {
                    responseModalities: ["AUDIO"],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: "Fenrir" } } }
                }
            }
        };
        webSocket.send(JSON.stringify(initialConfig));

        // Start Microphone
        log("Requesting Microphone Access...");
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
            microphone = audioContext.createMediaStreamSource(stream);
            
            // Capture audio chunks
            processor = audioContext.createScriptProcessor(4096, 1, 1);
            microphone.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const base64Data = encodePcm16(inputData);
                
                if (webSocket.readyState === WebSocket.OPEN) {
                    webSocket.send(JSON.stringify({
                        realtimeInput: { mediaChunks: [{ mimeType: "audio/pcm;rate=16000", data: base64Data }] }
                    }));
                }
            };
            log("Microphone Active. Speak directly to the Swarm.");
        } catch (e) {
            log(`Audio Error: ${e.message}`);
        }
    };

    webSocket.onmessage = async (e) => {
        let msg;
        if (e.data instanceof Blob) {
            // Google sends a JSON string inside a Blob/ArrayBuffer
            const text = await e.data.text();
            msg = JSON.parse(text);
        } else {
            msg = JSON.parse(e.data);
        }

        // Handle Server Content (Audio execution)
        if (msg.serverContent && msg.serverContent.modelTurn) {
            msg.serverContent.modelTurn.parts.forEach(part => {
                if (part.inlineData) {
                    audioBufferQueue.push(part.inlineData.data);
                    processAudioQueue();
                } else if (part.text) {
                    log(part.text, "ai");
                }
            });
        }
        
        // Handle Tool Calls (Preventing Network Halts)
        if (msg.toolCall) {
            msg.toolCall.functionCalls.forEach(async (call) => {
                log(`[Network Diagnostics] Intercepted Tool Request: ${call.name}()`);
                try {
                    let toolRes;
                    if (call.name === "get_current_market_status") {
                        log("Swarm triggered internal tool check. Querying backend...");
                        matrixStatus.innerText = "ACTIVE CLUSTERING (CALCULATING...)";
                        matrixStatus.classList.add("pulse");
                        
                        const res = await fetch('/api/market-status');
                        if (!res.ok) {
                            throw new Error(`HTTP Error Status: ${res.status}`);
                        }
                        
                        // Parse safely
                        toolRes = await res.json();
                        log("Quantitative Data retrieved from Python Backend.");
                        
                        // Dynamically update the Glassmorphic UI Widgets in Real-Time!
                        if (toolRes.gaussian_risk) {
                            document.getElementById('gauss-widget').innerText = toolRes.gaussian_risk;
                            document.getElementById('student-widget').innerText = toolRes.student_t_risk;
                            
                            setTimeout(() => {
                                matrixStatus.innerText = `LIVE VIX: ${toolRes.vix_live} (1.0X SPY)`;
                                matrixStatus.classList.remove("pulse");
                            }, 2000);
                        } else {
                            setTimeout(() => {
                                matrixStatus.innerText = "MARKET ALLOCATION STABILIZED";
                                matrixStatus.classList.remove("pulse");
                            }, 2000);
                        }
                        
                    } else {
                        log(`AI attempted to inject an unauthorized tool: ${call.name}(). Intercepted.`);
                        toolRes = { error: "Operation is restricted. Tool execution fallback triggered." };
                    }

                    // Force a response back to Google no matter what to prevent a 1008 Socket Crash
                    const toolResponsePayload = {
                        toolResponse: {
                            functionResponses: [{ 
                                id: call.id, 
                                name: call.name, 
                                response: toolRes 
                            }]
                        }
                    };
                    
                    log("Transmitting Quantitative Tool Payload back to Google...");
                    webSocket.send(JSON.stringify(toolResponsePayload));
                    log("Payload accepted. Resuming Swarm Dialogue...");
                    
                } catch (err) {
                    log(`[FATAL JS ERROR during Tool execution]: ${err.message}`, "system");
                    // Send an emergency fallback so the AI doesn't hang forever
                    const emergencyFallback = {
                        toolResponse: {
                            functionResponses: [{ 
                                id: call.id, 
                                name: call.name, 
                                response: { error: `Internal UI Exception: ${err.message}` } 
                            }]
                        }
                    };
                    webSocket.send(JSON.stringify(emergencyFallback));
                }
            });
        }
    };

    webSocket.onclose = (e) => {
        log(`WebSocket Connection Closed. Code: ${e.code}. Reason: ${e.reason || "Unknown"}`);
        disconnect();
    };
    webSocket.onerror = (e) => log(`WebSocket Error encountered.`);
}

function disconnect() {
    if (webSocket) webSocket.close();
    if (processor) processor.disconnect();
    if (microphone) microphone.disconnect();
    if (audioContext) audioContext.close();
    
    statusDot.className = "dot disconnected";
    statusText.innerText = "OFFLINE";
    connectBtn.classList.remove("hidden");
    disconnectBtn.classList.add("hidden");
    log("Uplink Severed.");
}

connectBtn.addEventListener('click', connectToSwarm);
disconnectBtn.addEventListener('click', disconnect);
