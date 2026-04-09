import os
import time
import json
import uuid
import logging
from typing import List, Dict

try:
    from openai import AsyncOpenAI
    has_openai = True
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-mock-key"))
except ImportError:
    has_openai = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] AutoSkillDistiller - %(message)s')

LOG_FILE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "proxy_logs.jsonl"
)

SKILLS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "memory", "skills"
)

os.makedirs(SKILLS_DIR, exist_ok=True)

class AutoSkillDistiller:
    def __init__(self, run_interval: int = 3600):
        self.run_interval = run_interval
        
    def load_traces(self) -> List[Dict]:
        """Loads interaction traces from the proxy logs"""
        if not os.path.exists(LOG_FILE_PATH):
            return []
            
        traces = []
        with open(LOG_FILE_PATH, 'r') as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return traces

    def detect_correction_loops(self, traces: List[Dict]) -> List[List[Dict]]:
        """
        Heuristic algorithm to detect when an agent failed at an action,
        was corrected by the user/system, and subsequently succeeded.
        """
        correction_loops = []
        # In a real scenario, this would group traces by session/trace_id
        # and search for "error" -> "user correction" -> "success" patterns
        # For this prototype, we group the most recent pairs.
        
        # Mocking detection of a correction loop:
        if len(traces) >= 2:
            correction_loops.append(traces[-2:])
        
        return correction_loops

    async def distill_skill(self, loop: List[Dict]) -> str:
        """
        Uses an LLM judge to extract the generalized rule from a correction loop.
        """
        if not has_openai:
            logging.warning("OpenAI library not found. Falling back to rule-based mock distillation.")
            return f"Always verify data types before division operations. Extracted from trace {loop[0].get('trace_id')}."
            
        try:
            # Format the loop into a transcript
            transcript = "\n".join([json.dumps(t.get("request_payload", {})) for t in loop])
            
            prompt = (
                "You are an Auto-Distillation AI. Review the following failure-to-success "
                "agent interaction trace. Extract a highly concise markdown skill rule that "
                "prevents this failure from happening again. Format as a markdown file.\n\n"
                f"TRACE:\n{transcript}"
            )
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Failed to compile skill via LLM: {str(e)}")
            return "Skill compilation failed due to network error."

    def write_skill_file(self, content: str):
        """Saves the distilled skill to the memory/skills directory."""
        filename = f"skill_distilled_{uuid.uuid4().hex[:8]}.md"
        filepath = os.path.join(SKILLS_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        logging.info(f"Successfully minted new skill configuration: {filepath}")

    async def run_daemon(self):
        logging.info("Auto Skill Distiller Active. Waiting for proxy traces...")
        while True:
            traces = self.load_traces()
            loops = self.detect_correction_loops(traces)
            
            if loops:
                logging.info(f"Detected {len(loops)} potential skill correction loops. Assaying...")
                for loop in loops:
                    skill_content = await self.distill_skill(loop)
                    if skill_content and "failed" not in skill_content.lower():
                        self.write_skill_file(skill_content)
                
                # Truncate logs or archive them so we don't double count
                if os.path.exists(LOG_FILE_PATH):
                    os.rename(LOG_FILE_PATH, LOG_FILE_PATH + f'.distill_{int(time.time())}.bak')
            
            time.sleep(self.run_interval)

if __name__ == "__main__":
    import asyncio
    distiller = AutoSkillDistiller(run_interval=60)
    asyncio.run(distiller.run_daemon())
