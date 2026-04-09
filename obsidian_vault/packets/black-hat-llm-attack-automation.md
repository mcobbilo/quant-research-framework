# Packet: Black-Hat Autonomous Exploitation (Carlini)

- packet_id: security-carlini-blackhat-llms
- source: Nicholas Carlini (Anthropic) - [un]prompted 2026
- confidence: high
- status: candidate
- tags: doctrine:execution_safety, cybersecurity, red_teaming, strix

Claim
Large Language Models have crossed the threshold from merely *assisting* human threat actors (writing isolated scripts) to autonomously orchestrating and executing complex, multi-step cyberattacks (the entire kill chain) without human-in-the-loop intervention.

Mechanism
Because modern agentic architectures (like LangGraph or Temporal) grant LLMs persistent memory, REPL terminal execution capabilities, and loop-routing, a malicious (or hallucinating) agent can autonomously perform reconnaissance, discover zero-day vulnerabilities in a codebase, write the exact exploit payload, and deploy it across a network.

Boundary
The autonomous kill chain is completely severed if the execution layer is strictly sandboxed. If an LLM is denied physical access to terminal execution (`SafeToAutoRun=False`) or bound by strict Pydantic execution exceptions that crash the script on unauthorized commands, the automation fails entirely.

Contradiction
The traditional cybersecurity belief is that AI is primarily a "defensive" scaling tool (e.g., auto-scanning logs for anomalies); Carlini's Black-Hat LLM doctrine proves that the exact same agentic capabilities provide exponentially greater scaling advantages to offensive actors constructing zero-day swarm attacks.
