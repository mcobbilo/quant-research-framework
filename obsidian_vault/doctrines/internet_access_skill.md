---
description: How the LangGraph Framework natively accesses the internet via CLI scaffolding
tags: [agent-reach, internet, bird, yt-dlp, exa]
---

# Agent-Reach Scaffolding Skill

**Path**: `src/tools/agent_reach.py`

This skill defines how secondary models and the orchestrator text-agents actively bypass tabular `.sqlite` data to read raw, unstructured human sentiment directly off the internet.

## Rule 1: The `bird` CLI Paradigm

Never attempt to write Python `requests` wrappers for Twitter/X. The platform will 403-block the agent instantly. Instead, we use the `@steipete/bird` NPM tool.

1. The human user must export their browser cookie via `Cookie-Editor`.
2. The LangGraph framework executes `npx bird search $SPY` in a subprocess.
3. The resulting stdout is piped directly into the **Macro Chief's** `AgentState`.

## Rule 2: Video Stream Decapitation

Whenever a live video URL is provided (e.g., Jerome Powell speaking on YouTube), do not use heavy visual multi-modal transcription. Use `yt-dlp --dump-json`.
This intercepts the core metadata, live subtitle strings, and description context instantly, converting a gigabyte video stream into a 5-kilobyte JSON string payload for the text-agent.

## Rule 3: Breaking News via MCP

Do not write custom beautifulsoup scrapers for Google News.
We natively rely on **Model Context Protocol (MCP)** execution bridges (e.g., `mcporter call 'exa.search()'`) to perform zero-cost, semantic vector searches for global macroeconomic events.

**Failure Mechanism:** If any of these 3 tools fail (Timeout, 403, or invalid JSON), the Python interface MUST catch the error and return a strict `[Tool Exception]` string. LangGraph agents must treat this string as "Neutral" confidence and rely entirely on mathematical primitives.
