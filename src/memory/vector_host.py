import os
import time
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class VectorHost:
    """
    Infinite-horizon semantic persistent memory for the Council Orchestrator.
    Logs agent failures and insights as dense embeddings instead of scrolling text arrays.
    """
    
    def __init__(self, data_path: str = "data/chroma", collection_name: str = "council_of_winners"):
        # Guarantee directory creation natively
        os.makedirs(data_path, exist_ok=True)
        
        # Initialize an embedded persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=data_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or fetch the core knowledge collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def cache_insight(self, insight: str, market_regime: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Stores an actionable mathematical failure or success into absolute memory.
        The ID is naturally generated.
        """
        # Augment specific metadata to allow structured retrieval
        meta = metadata or {}
        meta.update({"market_regime": market_regime, "timestamp": int(time.time())})
        
        doc_id = f"insight_{int(time.time()*100)}"
        
        self.collection.add(
            documents=[insight],
            metadatas=[meta],
            ids=[doc_id]
        )
        return doc_id
        
    def retrieve_similar_regimes(self, query: str, regime: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        When the Agent trains a new XGBoost ensemble in a 'High Volatility' regime,
        it calls this function to explicitly retrieve previous errors specific to that type.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            # Implicitly filter so COVID-19 embeddings aren't retrieved during 2026 bull markets
            where={"market_regime": regime}
        )
        
        # Formatting output for standard JSON Swarm ingestion
        if not results['documents']: return []
        
        payload = []
        for doc_arr, meta_arr, dist_arr in zip(results['documents'], results['metadatas'], results['distances']):
            for doc, meta, distance in zip(doc_arr, meta_arr, dist_arr):
                payload.append({
                    "insight": doc,
                    "metadata": meta,
                    "similarity_distance": distance
                })
                
        return payload
        
if __name__ == "__main__":
    # Test execution boundary
    memory = VectorHost(data_path="data/chroma")
    print("Injecting catastrophic trial #42 failure...")
    memory.cache_insight(
        "XGBoost depth over-fit due to rolling lookahead bias in VPIN calculation. T-stats inverted.",
        market_regime="high_volatility",
        metadata={"strategy": "vpin_ensemble"}
    )
    
    print("\n[+] Success. Persisted to disk.")
    retrieval = memory.retrieve_similar_regimes(
        query="Why did the XGBoost fail during high volatility?",
        regime="high_volatility"
    )
    print("\nRetrieval test:", retrieval)
