from crewai import Agent
import subprocess
import json
import time
import logging

ANTI_HALLUCINATION_PROTOCOL = """
### SYSTEMIC ZERO-HALLUCINATION GUARDRAILS ###
You are operating within a strictly audited Quantitative Execution Framework. You MUST adhere to the following 6 Protocols:

1. THE REFUSAL PROTOCOL:
If at any point you are uncertain about a fact, stop and write: UNCERTAIN: [what you don't know]
If you cannot complete any part of the request accurately, write: CANNOT COMPLETE: [specific reason]
Never fill gaps with assumptions. Incomplete and honest beats complete and wrong.

2. THE CONFIDENCE SCORER:
After completing your response, score every factual claim:
[HIGH] - You would stake your reputation on this
[MEDIUM] - You believe this but recommend verifying
[LOW] - This is your best guess, treat with caution

3. THE SOURCE CHALLENGER:
For every key claim, cite the type of source, the time period, and your confidence level. If you can't source it, flag it.

4. THE ASSUMPTION AUDIT:
Before asserting a conclusion, list every assumption you are making about the underlying data. Formally validate that these assumptions do not violate our strict No-Lookahead Bias rule.

5. THE BOUNDARY SETTER:
- Never invent statistics without flagging them
- Never suggest actions that bypass Risk Management boundaries
- Never present opinions as facts
- Never skip steps to save space

6. THE RED TEAM PASS:
Before outputting your final response, switch roles internally to a hostile critic whose job is to find every hallucination, exaggeration, and unsupported claim in your own logic. Fix or remove every flagged section before producing the final output.
"""

def fetch_mcp_prompt(prompt_name, fallback_prompt):
    """
    Subprocess JSON-RPC wrapper to pull State-of-the-Art prompt configurations dynamically
    from the prompts.chat MCP server. Includes graceful degradation.
    """
    try:
        process = subprocess.Popen(
            ["npx", "-y", "prompts.chat", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Give the JS runtime a brief second to initialize
        time.sleep(1)
        
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": "prompts/get",
            "id": 1,
            "params": {"name": prompt_name}
        }) + "\n"
        
        process.stdin.write(request)
        process.stdin.flush()
        
        response_data = process.stdout.readline()
        process.terminate()
        
        if response_data:
            response = json.loads(response_data)
            # Check if the MCP server routed an HTTP 404 internally
            if "error" in response:
                logging.warning(f"MCP Server Error [{prompt_name}]: {response['error'].get('message', 'Unknown Error')}")
                return fallback_prompt
                
            if "result" in response and "messages" in response["result"]:
                fetched_text = response["result"]["messages"][0]["content"]["text"]
                logging.info(f"Successfully bound MCP Prompt Template: {prompt_name}")
                return fetched_text
                
    except Exception as e:
        logging.warning(f"MCP Node Runtime Exception: {e}")
        
    return fallback_prompt

def load_agency_template(filename: str, legacy_fallback: str) -> str:
    """Loads the massive XML Specialist logic from the cloned agency-agents repository."""
    import os
    file_path = f"/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/agents/prompts/{filename}"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return legacy_fallback

def create_macro_chief():
    return Agent(
        role='Macro Intelligence Chief',
        goal='Aggregate and correlate top-level economic data from open-source feeds.',
        backstory=fetch_mcp_prompt("Data Scientist", load_agency_template("macro_chief.md", 'An elite intelligence officer...')) + "\n" + ANTI_HALLUCINATION_PROTOCOL,
        allow_delegation=False,
        verbose=True
    )

def create_quant_developer():
    return Agent(
        role='Quant Developer',
        goal='Maintain and optimize the TimesFM model architectures and pipeline logic.',
        backstory=fetch_mcp_prompt("Python Architect", load_agency_template("quant_developer.md", 'An expert software engineer...')) + "\n" + ANTI_HALLUCINATION_PROTOCOL,
        allow_delegation=False,
        verbose=True
    )

from langchain.tools import tool
import sys
import os

@tool("Search quantitative encyclopedia strategies")
def search_trading_strategies(query: str) -> str:
    """
    Query the 151 Trading Strategies encyclopedia using the ReMe Vector Architecture. 
    Pass a thematic string and receive the extracted procedural memory parameters.
    """
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.memory.local_persistence import ReMeVectorEngine
    try:
        engine = ReMeVectorEngine()
        memories = engine.retrieve_memory_sync(query=query, name="auto_research_agent")
        
        # If ReMe hasn't ingested vectors yet or lacks API keys, provide fallback data mapping
        if "Awaiting" in str(memories):
            return f"ReMe Initialization State: {memories}\n\nFallback: Proceed with core statistical modeling."
            
        output = "ReMe Procedural Strategies Extracted:\n\n"
        for mem in memories:
            output += f"--- Vector ID [{mem.memory_id}] ---\n{mem.memory_content}\n\n"
            
        return output
    except Exception as e:
        return f"ReMe Procedural Search Failed: {e}"

def create_auto_research_scientist():
    return Agent(
        role='Auto-Research Scientist',
        goal='Formulate novel quantitative hypotheses and execute TimesFM experiments.',
        backstory=fetch_mcp_prompt("Research Scientist", load_agency_template("auto_research_scientist.md", 'A brilliant statistician...')) + "\n" + ANTI_HALLUCINATION_PROTOCOL,
        allow_delegation=False,
        tools=[search_trading_strategies],
        verbose=True
    )

def create_risk_manager():
    return Agent(
        role='Execution & Risk Manager',
        goal='Execute validated trades through the OpenAlice framework while enforcing strict risk guardrails.',
        backstory=fetch_mcp_prompt("Risk Manager", load_agency_template("risk_manager.md", 'A hardened risk controller...')) + "\n" + ANTI_HALLUCINATION_PROTOCOL,
        allow_delegation=False,
        verbose=True
    )

