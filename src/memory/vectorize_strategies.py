import json
import numpy as np
import os
import sys

# Standardize path imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.memory.local_persistence import LocalMemoryStore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

def vectorize():
    print("[RAG] Loading 384-D Semantic Vector Network (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open('/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/memory/strategies_rag/strategies_rag_corpus.json', 'r') as f:
        corpus = json.load(f)
        
    print(f"[RAG] Successfully Loaded {len(corpus)} Physical Quant Documents.")
    
    strategy_db_path = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/memory/strategies_index.json"
    memory = LocalMemoryStore(db_path=strategy_db_path)
    
    print("[RAG] Passing text embeddings into 1-Bit TurboQuant QJL Reducer...")
    for chunk in corpus:
        text = chunk['title'] + " " + chunk['content']
        raw_embed = model.encode(text)
        memory.insert_vector_memory(chunk['title'], raw_embed)
        
    print("[RAG] Writing compressed matrix arrays to disk...")
    memory.save_vector_index()
    
    print("[RAG] Validating Search Integrity...")
    q_vec = model.encode("Volatility Arbitrage")
    res = memory.search_memory(q_vec, top_k=3)
    print("Top 3 Hits for 'Volatility Arbitrage':", res)

if __name__ == '__main__':
    vectorize()
