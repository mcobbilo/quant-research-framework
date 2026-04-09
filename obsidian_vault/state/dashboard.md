# Quant Framework - Memory Dashboard

## Current State (Last Updated: 2026-03-28)
- **Total Packets:** 4
  - *Promoted:* 4
  - *Candidate:* 0
  - *Drafts:* 0
- **Doctrines:** 1

## Coverage Matrix
- `doctrine:execution_safety`: Deep (3 packets)
- `doctrine:context_compression`: Intermediate (1 packet)

## Top Knowledge Gaps
1. Real-World Execution Slippage (No current live Alpaca data).
2. T10YFF Macro Context Injection edge cases.

## Recent Architectural Decisions
- [2026-03-28] Transition to Temporal Execution: Bounded critical Alpaca trade loops directly inside `@workflow.defn` to prevent script-crash vulnerabilities mid-transaction.
