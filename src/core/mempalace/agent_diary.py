import os
import datetime

class AgentDiary:
    """
    Independent Specialist Memory structure following MemPalace principles.
    Each agent maintains its own chronological log that prevents cross-contamination.
    """
    def __init__(self, agent_name, diary_dir="src/core/mempalace/diaries"):
        self.agent_name = agent_name
        self.diary_dir = diary_dir
        os.makedirs(self.diary_dir, exist_ok=True)
        self.diary_path = os.path.join(self.diary_dir, f"{self.agent_name}.log")

    def write(self, entry):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(self.diary_path, "a") as f:
            f.write(f"[{timestamp}] {entry}\n")

    def read(self, last_n=10):
        if not os.path.exists(self.diary_path):
            return ""
            
        with open(self.diary_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        if not lines:
            return ""
            
        return "\n".join(lines[-last_n:])
