---
description: Constraints and Rules for Bilevel Swarm Autoresearch 
tags: [bilevel, autoresearch, meta-agent, code-generation]
---

# Bilevel Evolution Doctrine

**Path**: `src/agents/bilevel_meta_agent.py` & `src/experimental/sandbox_inner_loop.py`

This doctrine defines how Meta-Agents (Outer Loops) are allowed to functionally rewrite the base codebase (Inner Loops) dynamically at runtime.

## Rule 1: Rigid Subprocess Isolation

The LLM is absolutely forbidden from writing changes directly into the `quant_framework/` executing memory. All Python logic it generates MUST be routed as strings, passed into `.py` files inside the sandbox environment, and executed purely via isolated `subprocess.run()`.

## Rule 2: Security Sanitization

Before the File Injection Engine inserts an LLM-generated vector string, it must explicitly scan for catastrophic filesystem primitives.
Blocked Python Keywords in generated code:

1. `os.remove` / `os.system`
2. `subprocess`
3. `__import__` (unless explicit quant libraries)
4. `open(`

If the Meta-Agent generates any of the above, immediately default to `pass` and trigger a negative `-999.0` Sharpe penalty to physically force the neural network gradients to avoid rewriting operating system states.

## Rule 3: Pandas Exclusivity

To optimize the 10,000+ simulation cycles required by bilevel research, inner loops cannot use complex iteration (`for i in range(len(df))`). The agent must perfectly utilize `pandas` native array manipulation (`np.where`, `.shift()`, `.rolling().mean()`). Fast array compute allows the Swarm loop to execute simultaneously alongside the master deep learning agent.
