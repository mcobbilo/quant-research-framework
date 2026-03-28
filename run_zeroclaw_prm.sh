#!/bin/bash
echo "=== ZeroClaw Quantitative PRM Initialization ==="

# Generate strict macOS Egress Sandbox profile
cat << 'EOF' > /tmp/agent_firewall.sb
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote tcp "*:443")) ; Strict API HTTPS requirement
(allow network-outbound (remote tcp "localhost:*")) ; Native Daemon communication
(allow network-outbound (remote tcp "127.0.0.1:*"))
EOF

# Boot the Rust Binary in the background
echo "[1] Booting ZeroClaw Rust Backend..."
"/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/zeroclaw/target/release/zeroclaw" agent &
ZEROCLAW_PID=$!

# Boot the Python Asynchronous PRM Loop under STRICT Physical Quarantine
echo "[2] Sinking Python Process Reward Model (PRM) Listener under macOS Egress Sandbox..."
sandbox-exec -f /tmp/agent_firewall.sb python3 "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/training/zero_claw_rl_loop.py" &
PYTHON_PID=$!

echo "=== System Live ==="
echo "ZeroClaw PID: $ZEROCLAW_PID"
echo "Python PRM (Sandbox Jailed) PID: $PYTHON_PID"
echo "To shutdown, run: kill $ZEROCLAW_PID $PYTHON_PID"

wait $ZEROCLAW_PID $PYTHON_PID
