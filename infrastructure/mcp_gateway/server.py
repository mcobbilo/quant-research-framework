from mcp.server.fastmcp import FastMCP
import subprocess
import os

# Define the absolute root of the quantitative framework to mount into the sandbox
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# Initialize the Sandbox Gateway MCP Server
mcp = FastMCP("Docker Sandbox MCP Gateway")

@mcp.tool()
def sandboxed_terminal_exec(command: str) -> str:
    """
    Executes a shell command inside an ephemeral, isolated Docker sandbox.
    WARNING: The /workspace directory (your quantitative framework) is mounted READ-ONLY.
    You cannot permanently delete, modify, or save files to the source directory.
    Output is returned via STDOUT/STDERR.
    """
    # 1. We construct the Docker command
    # --rm: Destroy container after execution
    # -v: Mount the project root to /workspace as read-only (ro)
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-w", "/workspace",
        "quant-agent-sandbox:latest",
        "sh", "-c", command
    ]

    try:
        # 2. Run the ephemeral container with an isolated timeout
        # Using a restricted timeout context prevents hanging the main orchestration layer.
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=120  
        )
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        output = []
        if stdout:
            output.append(f"STDOUT:\n{stdout}")
        if stderr:
            output.append(f"STDERR:\n{stderr}")
            
        if not output:
             output.append(f"Command exited with return code {result.returncode} (No Output).")
             
        return "\n\n".join(output)
        
    except subprocess.TimeoutExpired as e:
        return f"Sandbox execution timed out after 120s. STDOUT:\n{e.stdout}"
    except Exception as e:
        return f"Sandbox architecture failure: {str(e)}"

if __name__ == "__main__":
    # Start the FastMCP gateway on stdio
    # The agent will connect to this standalone server, fully decoupled from mathematical routing logic
    mcp.run()
