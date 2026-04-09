Packet: Pydantic Schema Injection

- packet_id: execution-llm-pydantic-parsing
- source: Phase 141 Final Routing
- confidence: high
- status: promoted
- tags: doctrine:execution_safety, schemas, prompt_engineering

Claim
Routing autonomous outputs by string-parsing LLM responses requires explicitly constraining the prompt generation window to a rigid, deterministic `json_schema` using typing frameworks.

Mechanism
By injecting `response_schema` bounds wrapped inside native Pydantic SDK blocks (`decision: Literal["LONG", "NEUTRAL"]`), the API endpoint halts text token predictions the instant the model tries generating conversational fluff, forcing it instead to only spit out key-value pairs that perfectly align with execution pipeline inputs.

Boundary
This logic fails entirely on earlier LLM models that lack native strict JSON grammar generation tools (e.g., GPT-3, Claude v1). 

Contradiction
"Just add 'respond only with JSON' to the system prompt." Prompt engineering is statistically imperfect for execution safety. Schema enforcement occurs natively in the backend API router, not as a natural language suggestion.
