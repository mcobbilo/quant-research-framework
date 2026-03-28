import os
import json

def initialize_subconscious():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sub_dir = os.path.join(base_dir, 'subconscious')
    
    os.makedirs(sub_dir, exist_ok=True)
    
    blocks = {
        "core_directives.md": "# Subconscious Block 1: Core Directives\nStrict, inviolable system execution rules.\n",
        "guidance.md": "# Subconscious Block 2: Guidance\nObservational, non-intrusive analytical advice parsed from previous anomalies.\n",
        "user_preferences.md": "# Subconscious Block 3: User Preferences\nExplicit mandates from the user (e.g., '1.0x Maximum Sizing', 'No Margin').\n",
        "project_context.md": "# Subconscious Block 4: Project Context\nCurrent architectural state of the quantitative matrix.\n",
        "session_patterns.md": "# Subconscious Block 5: Session Patterns\nIdentified mathematical anomalies or recurring data glitches.\n",
        "pending_items.md": "# Subconscious Block 6: Pending Items\nUpcoming pipeline executions or structural TODOs.\n",
        "self_improvement.md": "# Subconscious Block 7: Self Improvement\nInternal crash post-mortems and RL ZeroClaw feedback.\n",
        "tool_guidelines.md": "# Subconscious Block 8: Tool Guidelines\nSpecific parameter boundaries for local Python execution APIs and Broker endpoints.\n"
    }
    
    for filename, content in blocks.items():
        filepath = os.path.join(sub_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"[Subconscious Router] Initialized memory block: {filename}")
        else:
            print(f"[Subconscious Router] Memory block {filename} already exists. Bypassing.")
            
    print(f"\n[Subconscious Router] All 8 Letta-AI memory blocks actively staged at: {sub_dir}")

def route_insight(block_name: str, insight: str):
    """
    Dynamically appends a new insight directly into the specified Letta AI persistent memory block.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, 'subconscious', block_name)
    
    if not os.path.exists(filepath):
        print(f"[Error] Memory block {block_name} does not exist.")
        return
        
    with open(filepath, 'a') as f:
        f.write(f"\n- {insight}\n")
    print(f"[Subconscious Router] Merged insight onto {block_name}")

if __name__ == "__main__":
    initialize_subconscious()
