import sqlite3
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

class JepaMcpServer:
    """
    Model Context Protocol (MCP) Server for the J-EPA Framework.
    Decouples raw SQL from Agent Logic by exposing 'Tools' for market context.
    Inspired by block/goose modularity.
    """
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        
    def _execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """ Thread-safe interaction with the market_data.db """
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            print(f"[MCP ERROR] Query failed: {e}")
            return pd.DataFrame()

    def get_market_context(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """
        Tool: Retrieve the multi-domain feature set for a specific ticker window.
        """
        query = f"SELECT * FROM market_features WHERE ticker = ? AND date BETWEEN ? AND ?"
        df = self._execute_query(query, (ticker, start_date, end_date))
        
        if df.empty:
            return {"status": "error", "message": f"No data found for {ticker}"}
            
        return {
            "status": "success",
            "ticker": ticker,
            "count": len(df),
            "data": df.to_dict(orient='list'),
            "domains": self._detect_domains(df.columns)
        }

    def fetch_latent_trace(self, hypothesis_id: str) -> Optional[pd.DataFrame]:
        """
        Tool: Get the historical attention weights and state for an Alpha hypothesis.
        Used by Agent Delta for 'Self-Healing' diagnostics.
        """
        query = "SELECT * FROM curiosity_logs WHERE hypothesis_id = ?"
        return self._execute_query(query, (hypothesis_id,))

    def get_risk_regime_vitals(self) -> Dict:
        """
        Tool: Get the high-level 'Vitals' of the current market regime (VIX, Skew, Bond Vol).
        Used by the GMM Brain to anchor soft clusters.
        """
        # Hardcoded vital tickers as the protocol standard
        vitals = ["^VIX", "^MOVE", "^SKEW"]
        query = f"SELECT date, ticker, close FROM market_prices WHERE ticker IN ({','.join(['?']*len(vitals))}) ORDER BY date DESC LIMIT 3"
        df = self._execute_query(query, vitals)
        return df.pivot(index='date', columns='ticker', values='close').to_dict()

    def get_architecture_graph(self) -> Dict:
        """
        Tool: High-fidelity Knowledge Graph of the Tri-Modal J-EPA.
        Exposes internal call chains and dependencies for Graph-RAG discovery.
        """
        graph = {
            "nodes": [
                {"id": "Brain", "type": "model", "file": "src/models/jepa_attention_engine.py", "desc": "Regime-Aware GMM Attention Core"},
                {"id": "Tools", "type": "bridge", "file": "src/core/mcp_data_server.py", "desc": "MCP Data & Skill Interface"},
                {"id": "Skills", "type": "logic", "file": "src/skills/base_skill.py", "desc": "Modular Quant Mathematical Operations"},
                {"id": "Auditor", "type": "agent", "file": "curiosity_engine.py", "desc": "Agent Delta Self-Healing Loop"},
                {"id": "Extractor", "type": "pipeline", "file": "src/experimental/jepa_extractor.py", "desc": "Feature Engineering & 20D Backtest Fork"}
            ],
            "edges": [
                {"source": "Auditor", "target": "Brain", "type": "stability_audit", "desc": "Delta monitors model weights for gradient explosion"},
                {"source": "Extractor", "target": "Tools", "type": "data_request", "desc": "Backtest fork calls MCP for microstructure context"},
                {"source": "Brain", "target": "Tools", "type": "skill_invocation", "desc": "Attention head selects 'VolTargeting' skill"},
                {"source": "Tools", "target": "Skills", "type": "bridge", "desc": "MCP server serves standalone skills as tools"}
            ]
        }
        return graph

    def _detect_domains(self, columns: List[str]) -> Dict[str, List[str]]:
        """ Maps raw columns to J-EPA Domains in real-time. """
        domain_map = {
            "CREDIT": [c for c in columns if any(k in c for k in ["HYG", "LQD", "credit", "spread"])],
            "VOLATILITY": [c for c in columns if any(k in c for k in ["VIX", "volatility", "ATR", "boll"])],
            "BREADTH": [c for c in columns if any(k in c for k in ["NYA200R", "breadth", "advance_decline"])],
            "MACRO": [c for c in columns if any(k in c for k in ["US10Y", "yield", "macro", "gravity"])]
        }
        return domain_map

# Global Singleton for ease of use across the Framework
MCP = JepaMcpServer()
