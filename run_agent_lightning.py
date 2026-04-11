import os
import time

from agentlightning import (
    Trainer,
    PromptTemplate,
    emit_reward,
    operation,
    rollout,
    set_active_tracer,
    OtelTracer,
    TraceToMessages,
)
from agentlightning.algorithm.apo import APO
from openai import AsyncOpenAI

from src.core.lightning_tasks import load_training_dataset, CuriosityTask
from curiosity_engine import (
    STRATEGY_NODE,
    OSINT_NODE,
    execute_and_evaluate,
    CHAMPION_SHARPE,
    get_database_schema,
)

# Initialize default Tracer to prevent '@operation' load-time crashes, and save it
# so we can pass it into the Trainer, preventing initialization mismatch.
GLOBAL_TRACER = OtelTracer()
set_active_tracer(GLOBAL_TRACER)

# Load Environment Variables
env_path = ".env"
env_vars = {}
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                env_vars[key] = val.strip('"').strip("'")

API_KEY = os.environ.get("XAI_API_KEY", env_vars.get("XAI_API_KEY"))
# Ensure the base URL maps cleanly to OpenAI client format
BASE_URL = os.environ.get("LLM_API_URL", "https://api.x.ai/v1")
if BASE_URL.endswith("/chat/completions"):
    BASE_URL = BASE_URL.replace("/chat/completions", "")

MODEL = os.environ.get("LLM_MODEL", "grok-4.20-reasoning")


# Define Initial Prompt for APO
INITIAL_PROMPT = PromptTemplate(
    template="""# Mathematical Strategy Hypothesis Generation
You are an institutional quant researcher. Develop a mathematically robust trading strategy hypothesis.

Inspirational Focus: {{ task.inspiration_seed }}

Ensure equations are clearly defined and logic operates free of lookahead biases. Provide concrete entry and exit rules.""",
    engine="jinja",
)


@operation
def generate_strategy(task: CuriosityTask, prompt: str) -> str:
    inspiration_seed = task["inspiration_seed"]

    # OSINT Verification pass
    osint_results = OSINT_NODE.execute(
        inspiration_seed, iteration_seed=inspiration_seed, mode="research"
    )
    digest = osint_results.get("digest", "No OSINT findings.")

    # Generate Strategy
    strat = STRATEGY_NODE.execute(
        iteration=1,
        memory="",
        knowledge=digest,
        inspiration=prompt,
        champion_info=f"Champion Sharpe: {CHAMPION_SHARPE}",
    )
    return strat["pitch"] if strat else ""


@operation
def write_and_eval_code(pitch: str) -> float:
    from curiosity_engine import CODER_NODE

    schema = get_database_schema()
    code = CODER_NODE.execute(pitch, schema, knowledge="")

    if not code:
        return -1.0

    result = execute_and_evaluate(code, f"Lightning_APO_{int(time.time())}")
    if result["status"] == "success":
        return result["sharpe"]

    return -1.0  # Penalize on runtime crash or failure to generate valid logic


@rollout
def curiosity_rollout(task: CuriosityTask, prompt_template: PromptTemplate) -> None:
    # Notice we pass `prompt_template.template` which APO dynamically modifies
    pitch = generate_strategy(task, prompt_template.template)
    if not pitch:
        emit_reward(-1.0)
        return

    sharpe = write_and_eval_code(pitch)
    emit_reward(sharpe)


from agentlightning import clear_active_tracer


def main():
    clear_active_tracer()

    print("==================================================")
    print(" CURIOSITY ENGINE v2.5 | AGENT LIGHTNING APO LOOP ")
    print("==================================================")

    dataset = load_training_dataset()

    # Subset to 2 items for 2-step verification phase to prove the RL loop works
    verification_dataset = dataset[:2]

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Initialize the Automatic Prompt Optimization Algorithm (APO)
    algorithm = APO[CuriosityTask](
        async_openai_client=client,
        gradient_model=MODEL,
        apply_edit_model=MODEL,
        beam_width=1,
        branch_factor=1,
        beam_rounds=1,  # 1 round of optimization for verification
        gradient_batch_size=1,
        val_batch_size=1,
    )

    # APO uses 'main_prompt' internally as its prompt identifier
    initial_resources = {"main_prompt": INITIAL_PROMPT}
    algorithm.set_initial_resources(initial_resources)

    # Initialize Trainer
    trainer = Trainer(
        tracer=GLOBAL_TRACER,
        algorithm=algorithm,
        adapter=TraceToMessages(),
        strategy="shm",
        n_runners=1,  # Single-thread for verification stability against API rates
    )

    print("[ACA] Starting APO Evaluation & Evolution Loop...")
    trainer.fit(
        agent=curiosity_rollout,
        train_dataset=verification_dataset,
        val_dataset=verification_dataset,
    )

    best_prompt = algorithm.get_best_prompt()
    print(f"\n[ACA] >> Best Optimized Template Discovered: <<\n{best_prompt.template}")


if __name__ == "__main__":
    main()
