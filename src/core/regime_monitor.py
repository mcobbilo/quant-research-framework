import numpy as np
import pandas as pd
import json

class RegimeMonitor:
    """
    Tracks and records the behavior of the Self-Organizing Council during 
    regime shifts (Genetic Divergence).
    """
    def __init__(self):
        self.history = []
        self.regime_shifts = []

    def record_step(self, t, price, ground_truth_regime, consensus_score, latent_divergence, predicted_prob):
        """
        Records a single measurement step.
        """
        self.history.append({
            't': t,
            'price': float(price),
            'regime': int(ground_truth_regime),
            'consensus': float(consensus_score),
            'divergence': float(latent_divergence),
            'prob': float(predicted_prob)
        })

    def analyze_recovery(self):
        """
        Calculates 'Recovery Time' following each regime shift.
        Recovery = Consensus score returning above 0.90 after a shift.
        """
        df = pd.DataFrame(self.history)
        df['regime_changed'] = df['regime'].diff().fillna(0) != 0
        shift_indices = df[df['regime_changed']].index.tolist()
        
        recovery_stats = []
        for shift_idx in shift_indices:
            new_regime = df.loc[shift_idx, 'regime']
            # Find when consensus recovers to > 0.90
            post_shift = df.loc[shift_idx:]
            recovered_idx = post_shift[post_shift['consensus'] > 0.90].index
            
            if len(recovered_idx) > 0:
                recovery_steps = recovered_idx[0] - shift_idx
            else:
                recovery_steps = -1 # Never recovered
                
            recovery_stats.append({
                'from_regime': df.loc[shift_idx-1, 'regime'] if shift_idx > 0 else -1,
                'to_regime': int(new_regime),
                'recovery_steps': int(recovery_steps)
            })
            
        return recovery_stats

    def save_report(self, path="genetic_divergence_report.json"):
        def to_native(obj):
            if isinstance(obj, (np.generic, np.ndarray)):
                return obj.item() if np.isscalar(obj) else obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj

        report = {
            'summary': to_native(self.analyze_recovery()),
            'raw_data': to_native(self.history)
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Phase 14 Report saved to {path}.")
