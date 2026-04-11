import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class OptionDealerAnalysis(BaseModel):
    direction: str = Field(description="Directional leaning: LONG, SHORT, or NEUTRAL")
    rationale: str = Field(
        description="Detailed rationale for the directional leaning based on options flow and heavy-tail risk"
    )


def get_option_dealer_llm():
    """
    Returns the LLM for the Option Dealer.
    Transitioned to xAI Grok (OpenAI-compatible endpoint) per user instructions
    to resolve memory bottlenecks with local 31B models.
    """
    raw_key = os.environ.get("XAI_API_KEY", "")
    clean_key = raw_key.strip('"').strip("'")

    model_name = os.environ.get("OPTION_DEALER_MODEL", "grok-4.2")

    return ChatOpenAI(
        model=model_name,
        base_url="https://api.x.ai/v1",
        api_key=clean_key,
        temperature=0.2,
        max_tokens=1024,
    )


from src.models.jamba_bridge import JambaBridge

# Global JambaBridge instance to avoid repeated GPU weight loading (MPS)
_jamba_bridge = None


def get_jamba_signal():
    global _jamba_bridge
    if _jamba_bridge is None:
        try:
            _jamba_bridge = JambaBridge()
        except Exception as e:
            print(f"[Option Dealer] Failed to initialize Jamba Bridge: {e}")
            return None
    return _jamba_bridge.get_latest_alpha_signal()


def option_dealer_node(state: dict) -> dict:
    """
    LangGraph node for evaluating options data via Nemotron-4-340b-instruct.
    Expects state to contain 'market_data' with options features (VIX, put/call ratio, etc).
    """
    try:
        llm = get_option_dealer_llm().with_structured_output(OptionDealerAnalysis)
        spy_data = state.get("spy_data", "No SPY telemetry provided.")
        vix_data = state.get("vix_data", "No VIX telemetry provided.")

        # Phase 26.2: HST-Jamba Alpha Infusion (Structural Sight)
        jamba_signal = get_jamba_signal()
        jamba_context = "No Jamba alpha signal available."
        if jamba_signal:
            jamba_context = f"""
            Regime Prediction: {jamba_signal["ensemble_prediction"]}
            Confidence: {jamba_signal["confidence"]:.2%}
            Class Probabilities: {jamba_signal["distribution"]}
            """

        # Phase 19: Structural Memory Injection (Structural Awareness)
        memory_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MEMORY_MAP.md"
        )
        structural_context = "No structural memory mapping found."
        if os.path.exists(memory_path):
            with open(memory_path, "r") as f:
                structural_context = f.read()

        prompt = f"""
        You are the Option Dealer. You analyze options flow, implied volatility surfaces, put/call ratios, and heavy-tail risk metrics focusing exclusively on the SPY ETF and VIX dynamics.
        Your job is to provide a highly specialized directional lean (LONG, SHORT, or NEUTRAL) for the current market state based on the provided index and volatility data.

        [HST JAMBA STRUCTURAL SIGHT]:
        {jamba_context}

        [STRUCTURAL MEMORY MAP]: 
        {structural_context}

        Current SPY Telemetry:
        {spy_data}

        Current VIX Telemetry:
        {vix_data}
        """

        messages = [
            SystemMessage(
                content="You are the Option Dealer for the autonomous quantitative trading desk. Output structured JSON."
            ),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        return {
            "option_dealer_direction": response.direction,
            "option_dealer_rationale": response.rationale,
        }
    except Exception as e:
        print(f"[Option Dealer Node] Error executing Nemotron inferrence: {e}")
        return {
            "option_dealer_direction": "NEUTRAL",
            "option_dealer_rationale": f"Inference failed with exception: {e}",
        }
