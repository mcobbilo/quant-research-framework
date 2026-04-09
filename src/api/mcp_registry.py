import os
from mcp.server.fastmcp import FastMCP

# Absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURIOSITY_PATH = os.path.join(BASE_DIR, "memory", "fractal", "CURIOSITY.md")
MEMORY_PATH = os.path.join(BASE_DIR, "memory", "fractal", "MEMORY.md")

# Create the MCP Server
mcp = FastMCP(
    "AutonomousCuriosityEngine",
    dependencies=["fastmcp", "pydantic"]
)

@mcp.resource("memory://curiosity_matrix")
def get_curiosity_matrix() -> str:
    """Returns the full Autonomous Curiosity Matrix."""
    if not os.path.exists(CURIOSITY_PATH):
        return "CURIOSITY.md not found."
    with open(CURIOSITY_PATH, "r") as f:
        return f.read()

@mcp.resource("memory://engine_logs")
def get_engine_logs() -> str:
    """Returns the most recent execution logs from MEMORY.md."""
    if not os.path.exists(MEMORY_PATH):
        return "MEMORY.md not found."
    with open(MEMORY_PATH, "r") as f:
        lines = f.readlines()
        return "".join(lines[-500:])

@mcp.tool()
def get_hypothesis_status(hypothesis_name: str) -> str:
    """
    Extracts the current pass [S] / fail [F] boolean and execution timestamp for a specifically requested hypothesis.
    """
    if not os.path.exists(CURIOSITY_PATH):
        return "Error: Matrix not found."
    with open(CURIOSITY_PATH, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if hypothesis_name.lower() in line.lower():
            if "[S]" in line:
                return f"Hypothesis '{hypothesis_name}' SUCCEEDED: {line.strip()}"
            elif "[F]" in line:
                return f"Hypothesis '{hypothesis_name}' FAILED: {line.strip()}"
            elif "[ ]" in line:
                return f"Hypothesis '{hypothesis_name}' is PENDING: {line.strip()}"
    return f"Hypothesis '{hypothesis_name}' not found."
