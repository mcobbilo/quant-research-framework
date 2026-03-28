import os
import sys
import uvicorn

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("CRITICAL: 'GEMINI_API_KEY' not found in environment. The Web Dashboard requires authentication.")
        sys.exit(1)
        
    print("\n[Swarm Dashboard] Booting FastAPI Server on port 8000...")
    print("[Swarm Dashboard] Open your browser to: http://localhost:8000\n")
    
    # Launch Uvicorn dynamically
    uvicorn.run(
        "src.interface.web_dashboard.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
