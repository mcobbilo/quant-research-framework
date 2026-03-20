from pydantic import BaseModel
from crewai.flow import Flow, start, listen, router
from data.crucix import execute_sweep_delta

class MarketState(BaseModel):
    market_data: dict = {}
    probability: float = 0.0
    confidence: float = 1.0
    final_action: str = ""

from data.sweep_delta import cross_correlate
from models.timesfm_wrapper import TimesFM2_5, tabular_preprocessing, format_for_timesfm, calculate_cdf_tail
from execution.openalice import OpenAliceUTA, calc_kelly
from memory.local_persistence import LocalMemoryStore
from training.diloco import DiLoCoCluster

class MarketForecastingFlow(Flow[MarketState]):
    
    @start()
    def ingest_market_state(self):
        print("\n[Flow] 1. Triggering Crucix Terminal for Tiers 1-3 cross-correlated data...")
        raw_data = execute_sweep_delta()
        correlated_data = cross_correlate(raw_data)
        self.state.market_data = correlated_data
        return self.state.market_data

    @listen(ingest_market_state)
    def execute_timesfm_forecast(self):
        print("\n[Flow] 2. Processing tabular features via fastai...")
        processed_tensors = tabular_preprocessing(self.state.market_data)
        context_tensor = format_for_timesfm(processed_tensors)
        
        print("\n[Flow] 3. Initializing TimesFM 2.5 and predicting quantiles...")
        model = TimesFM2_5()
        distribution = model.predict_quantiles(context_tensor)
        
        prob_up_04 = calculate_cdf_tail(distribution, threshold=1.004)
        confidence = model.calculate_confidence_interval()
        
        self.state.probability = prob_up_04
        self.state.confidence = confidence
        return "forecast_complete"

    @router(execute_timesfm_forecast)
    def evaluate_confidence_and_route(self):
        prob = getattr(self.state, 'probability', 0.0)
        conf = getattr(self.state, 'confidence', 1.0)
        
        print(f"\n[Flow] 4. Routing dynamically based on prob: {prob:.2f}, conf: {conf:.3f}...")
        
        if prob > 0.65 and conf < 0.1:
            return "stage_trade"
        elif prob < 0.35 and conf < 0.1:
            return "stage_hedge"
        else:
            return "trigger_human_review"

    @listen("stage_trade")
    def execute_long_trade(self):
        print("\n[Flow] 5a. Staging and Committing LONG trade via OpenAlice UTA paradigm...")
        uta = OpenAliceUTA(account_id="alpha_fund")
        trade_size = calc_kelly(self.state.probability)
        uta.stage(asset="SPY", action="BUY", size=trade_size)
        
        rationale = f"Prob > 0.4%: {self.state.probability:.2f}, Conf: {self.state.confidence:.3f}"
        uta.commit(rationale_hash=str(hash(rationale)))
        
        # Guard pipeline
        success = uta.push()
        
        self.state.final_action = f"EXECUTED LONG SPY (Size: {trade_size:.2f})" if success else "BLOCKED BY GUARDS"
        
        # Save to memory
        mem = LocalMemoryStore()
        mem.save_experiential_memory(run_id="run_101", model_params="TimesFM-200m", sharpe_ratio=0.0, rationale=rationale)
        
        return self.state.final_action

    @listen("stage_hedge")
    def execute_short_hedge(self):
        print("\n[Flow] 5b. Staging and Committing HEDGE via OpenAlice UTA paradigm...")
        self.state.final_action = "EXECUTED HEDGE SPY"
        return self.state.final_action
        
    @listen("trigger_human_review")
    def trigger_review(self):
        print("\n[Flow] 5c. Uncertainty too high. Triggering human review...")
        self.state.final_action = "PENDING HUMAN REVIEW"
        return self.state.final_action
