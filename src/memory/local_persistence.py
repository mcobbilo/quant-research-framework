import json
import os
import numpy as np
from datetime import datetime

class QJLCompressor:
    """
    Quantized Johnson-Lindenstrauss (QJL) Memory Transformer.
    Compresses 32-bit floating point high-dimensional vectors down to native 1-bit boolean hashes.
    """
    def __init__(self, input_dim=384, reduced_dim=256, seed=42):
        self.input_dim = input_dim
        self.reduced_dim = reduced_dim
        # Generate a deterministic Gaussian random projection matrix
        rng = np.random.RandomState(seed)
        self.projection_matrix = rng.randn(self.reduced_dim, self.input_dim) / np.sqrt(self.reduced_dim)

    def compress(self, vector: np.ndarray) -> np.ndarray:
        """
        Projects the vector into the reduced space and extracts exactly 1-bit (Sign).
        Returns a compressed boolean array (can be packed into bytes for ultra-low memory).
        """
        projected = np.dot(self.projection_matrix, vector)
        # 1-Bit Trick: Quantize to Sign Bit (+1 or -1, mapped to True/False for packing)
        return projected > 0

    def compute_similarity(self, qjl_a: np.ndarray, qjl_b: np.ndarray) -> float:
        """
        Calculates similarity using native bitwise XOR (Hamming Distance).
        The underlying mathematical distance is proportionally preserved.
        """
        hamming_distance = np.sum(qjl_a != qjl_b)
        similarity = 1.0 - (hamming_distance / self.reduced_dim)
        return similarity

class LocalMemoryStore:
    def __init__(self, db_path="memory_store.json"):
        self.db_path = db_path
        self.qjl = QJLCompressor()
        self.qjl_index = []
        self.qjl_metadata = []
        self._initialize_db()
        self.load_vector_index()

    def save_vector_index(self):
        idx_path = self.db_path.replace(".json", "_vector_matrix.json")
        with open(idx_path, "w") as f:
            json.dump({
                "metadata": self.qjl_metadata,
                "vectors": [np.array(v).tolist() for v in self.qjl_index]
            }, f)
            
    def load_vector_index(self):
        idx_path = self.db_path.replace(".json", "_vector_matrix.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r") as f:
                data = json.load(f)
                self.qjl_metadata = data["metadata"]
                self.qjl_index = [np.array(v, dtype=bool) for v in data["vectors"]]

    def _initialize_db(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({"factual": [], "experiential": [], "working": []}, f)

    def save_experiential_memory(self, run_id, model_params, sharpe_ratio, rationale):
        """Saves backtest results and model iterations for agent persistence."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "model_params": model_params,
            "performance_metric": sharpe_ratio,
            "rationale": rationale
        }
        
        with open(self.db_path, "r+") as f:
            data = json.load(f)
            data["experiential"].append(entry)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
            
            # Conditionally Trigger AutoDream Memory Consolidation periodically
            if len(data["experiential"]) % 5 == 0:
                print(f"[Memory] Experiential density threshold reached. Spawning background AutoDream pruning...")
                import subprocess
                script_path = os.path.join(os.path.dirname(__file__), "autodream.py")
                subprocess.Popen(["python3", script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        print(f"[Memory] Saved experiential learning for run: {run_id}")

    def insert_vector_memory(self, content_id, raw_vector: np.ndarray):
        """Passes a 32-bit raw embedding into the 1-bit TurboQuant compressor."""
        compressed_hash = self.qjl.compress(raw_vector)
        self.qjl_index.append(compressed_hash)
        self.qjl_metadata.append(content_id)
        
    def search_memory(self, raw_query_vector: np.ndarray, top_k=3):
        """Executes ultra-fast 1-bit hamming semantic search."""
        if not self.qjl_index:
            return []
            
        query_hash = self.qjl.compress(raw_query_vector)
        scores = [self.qjl.compute_similarity(query_hash, stored_hash) for stored_hash in self.qjl_index]
        
        # Sort by highest similarity
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.qjl_metadata[i], scores[i]) for i in ranked_indices]

import asyncio

class ReMeVectorEngine:
    """
    Agentic Memory Kit Integration (agentscope-ai/ReMe)
    Replaces static RAG retrievals with dynamic entity memories.
    """
    def __init__(self, working_dir="/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/memory/reme_db"):
        self.working_dir = working_dir
        
    async def _init_reme(self):
        from reme import ReMe
        
        # Configure ReMe Vector Backbone
        reme = ReMe(
            working_dir=self.working_dir,
            default_llm_config={
                "model_name": "gemini-3.1-pro-preview",
            },
            default_vector_store_config={
                "backend": "local", 
            }
        )
        await reme.start()
        return reme
        
    def add_procedural_memory_sync(self, content: str, name="auto_research_agent"):
        async def run():
            reme = await self._init_reme()
            await reme.add_memory(memory_content=content, user_name=name)
            await reme.close()
        try:
            asyncio.run(run())
        except Exception as e:
            print(f"[ReMe Integration] Vector ingestion gracefully bypassed: {e}")

    def retrieve_memory_sync(self, query: str, name="auto_research_agent", top_k=2):
        async def run():
            reme = await self._init_reme()
            # Dynamic vector search simulating procedural knowledge extraction
            memories = await reme.retrieve_memory(query=query, user_name=name)
            await reme.close()
            return memories
        try:
            return asyncio.run(run())
        except Exception as e:
            # Fallback if ReMe default embedding keys are not configured in environment
            return f"[ReMe Procedural System Alert] Awaiting Open-Source Embedding Configuration: {e}"
