import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, Any
from agents.option_dealer import option_dealer_node

class DebateState(TypedDict):
    spy_data: str
    vix_data: str
    macro_chief_analysis: Optional[str]
    risk_manager_analysis: Optional[str]
    option_dealer_direction: Optional[str]
    option_dealer_rationale: Optional[str]
    final_decision: Optional[str]

def macro_chief_node(state: DebateState) -> Dict[str, Any]:
    return {"macro_chief_analysis": "Evaluating broad market macro conditions. Base-case trend appears stable."}

def risk_manager_node(state: DebateState) -> Dict[str, Any]:
    return {"risk_manager_analysis": "Volatility looks bounded within historical norms. Capital allocation limits enforced."}

def synthesis_node(state: DebateState) -> Dict[str, Any]:
    # Basic synthesis incorporating the option dealer's decision as primary input
    option_dir = state.get("option_dealer_direction", "NEUTRAL")
    decision = option_dir if option_dir in ["LONG", "SHORT"] else "NEUTRAL"
    return {"final_decision": f"Consensus synthesized: {decision}"}

def build_debate_graph():
    graph = StateGraph(DebateState)
    
    graph.add_node("macro_chief", macro_chief_node)
    graph.add_node("option_dealer", option_dealer_node)
    graph.add_node("risk_manager", risk_manager_node)
    graph.add_node("synthesis", synthesis_node)
    
    # Linear flow for the debate scaffolding
    graph.set_entry_point("macro_chief")
    graph.add_edge("macro_chief", "option_dealer")
    graph.add_edge("option_dealer", "risk_manager")
    graph.add_edge("risk_manager", "synthesis")
    graph.add_edge("synthesis", END)
    
    return graph.compile()

def execute_debate() -> Dict[str, Any]:
    """Entry point used by execution/openalice.py to execute the LangGraph Syndicate."""
    debater = build_debate_graph()
    
    initial_state = {
        "spy_data": "SPY Current Price: 512.45, 1D Return: +0.2%, Order Flow: Neutral to Bullish",
        "vix_data": "VIX Current: 15.2, Term Structure: Contango",
    }
    
    return debater.invoke(initial_state)

if __name__ == "__main__":
    result = execute_debate()
    print("Debate Result:")
    print(result)
