import os
import sys
import logging
import subprocess
from watchfiles import watch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] HotReloader - %(message)s')

SRC_DIR = os.path.join(os.path.dirname(__file__), "..")

class LiveSelfModificationReloader:
    """
    Enables the MetaClaw self-modification loops.
    When the principal agent rewrites a .py file (e.g. optimizing an execution script),
    this daemon intercepts the filesystem event and gracefully restarts the background 
    Celery workers or other python sub-processes without dropping the main LLM websocket.
    """
    
    def __init__(self):
        self.worker_process = None
        
    def start_celery(self):
        """Starts the celery worker tracking the quant mathematical framework."""
        if self.worker_process:
            logging.info("Terminating existing Celery worker...")
            self.worker_process.terminate()
            self.worker_process.wait()
            
        logging.info("Booting Celery Worker...")
        # Since this executes from within quant_framework/src/core/ it needs correct pythonpath
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(SRC_DIR, ".."))
        
        self.worker_process = subprocess.Popen(
            [sys.executable, "-m", "celery", "-A", "src.api.celery_worker", "worker", "--loglevel=info"],
            env=env,
            cwd=os.path.abspath(os.path.join(SRC_DIR, ".."))
        )

    def run(self):
        self.start_celery()
        logging.info(f"Watching for Live Self-Modification events securely in {SRC_DIR}...")
        
        # watchfiles yields sets of changes
        for changes in watch(SRC_DIR):
            modified = False
            for change, path in changes:
                if path.endswith(".py"):
                    logging.info(f"Detected intelligent self-modification: {path}")
                    modified = True
                    break
            
            if modified:
                logging.info("Hot-reloading mathematical engine framework...")
                self.start_celery()

if __name__ == "__main__":
    reloader = LiveSelfModificationReloader()
    try:
        reloader.run()
    except KeyboardInterrupt:
        if reloader.worker_process:
            reloader.worker_process.terminate()
        logging.info("Hot Reloader terminated.")
