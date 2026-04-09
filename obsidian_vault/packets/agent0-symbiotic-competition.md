# Packet: Agent0 Symbiotic Tool Evolution

- packet_id: ai-research-agent0-symbiosis
- source: arXiv:2511.16043 (Agent0: Unleashing Self-Evolving Agents from Zero Data)
- confidence: high
- status: candidate
- tags: doctrine:self_evolution, multi_agent, tool_use, curriculum_learning

Claim
Agents can continuously evolve their reasoning and tool-use capabilities from a zero-data starting point by utilizing a dual-agent symbiotic competition rather than relying on human-curated datasets.

Mechanism
The architecture pits two instances of the same base model against each other: a Curriculum Agent continuously proposes increasingly difficult frontier algorithms/tasks, while an Executor Agent attempts to solve them using external tools; as the Executor masters tools, it forces the Curriculum Agent to generate exponentially harder problem sets.

Boundary
This self-reinforcing cycle is inherently bounded by the underlying capacity of the physical tools provided to the Executor; if the Executor's tools (e.g., Python execution, calculators) cannot logically solve the Curriculum Agent's hardest problems, the evolution loop halts.

Contradiction
Traditional RLHF (Reinforcement Learning from Human Feedback) assumes that scalable agent intelligence fundamentally requires massive human-annotated data scaling, while Agent0 proves algorithmic intelligence can evolve purely via adversarial self-play.
