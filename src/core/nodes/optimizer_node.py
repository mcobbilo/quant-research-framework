import os
import re
import numpy as np
import copy
from src.core.rlm_scaffold import RLMScaffold

class OptimizerNode:
    def __init__(self, harness_config, call_llm_func, execute_eval_func):
        self.config = harness_config
        self.call_llm = call_llm_func
        self.execute_evaluate = execute_eval_func
        
    def _extract_params_from_code(self, code):
        match = re.search(r'EGGROLL_PARAMS\s*=\s*(\{.*?\})', code, re.DOTALL)
        if match:
            try:
                # Safe evaluation of basic dictionaries
                return eval(match.group(1))
            except:
                return None
        return None

    def _inject_params_to_code(self, code, params):
        # We replace the EGGROLL_PARAMS dictionary in the string with the updated one
        def replacer(match):
            return f"EGGROLL_PARAMS = {params}"
        return re.sub(r'EGGROLL_PARAMS\s*=\s*\{.*?\}', replacer, code, flags=re.DOTALL)

    def execute(self, best_failed_code):
        print("[NODE] > OptimizerNode (EGGROLL ES Parameter Refinement)...")
        
        # 1. Parameter Lifting via Agent Epsilon (RLM)
        epsilon_info = self.config["agents"]["epsilon"]
        epsilon_rlm = RLMScaffold("Epsilon-Optimizer", self.call_llm)
        rlm_context = {
            "BASE_CODE": best_failed_code
        }
        rlm_prompt = f"{epsilon_info['system']}\n\n{epsilon_info['template']}\n\n" \
                     f"Rewrite the following code to implement EGGROLL_PARAMS:\n\n{best_failed_code}"
                     
        parameterized_code_raw = epsilon_rlm.run_repl(rlm_prompt, context_vars=rlm_context, max_iterations=3, temperature=0.1)
        parameterized_code = str(parameterized_code_raw).strip()
        
        # Strip markdown syntax if Epsilon added it
        match = re.search(r'```python(.*?)```', parameterized_code, re.DOTALL)
        if match:
            parameterized_code = match.group(1).strip()
            
        base_params = self._extract_params_from_code(parameterized_code)
        
        if not base_params or not isinstance(base_params, dict):
            print("[EGGROLL] Optimization Failed. Epsilon could not extract EGGROLL_PARAMS.")
            return best_failed_code, -1.0
            
        # 2. Setup Evolution Strategy
        POPULATION_SIZE = 5
        GENERATIONS = 3
        LEARNING_RATE = 0.5
        NOISE_STD = 0.10
        
        print(f"[EGGROLL] Commencing Evolution Strategy (N={POPULATION_SIZE}, G={GENERATIONS})")
        print(f"[EGGROLL] Initial Mean Params: {base_params}")
        
        current_mean_params = copy.deepcopy(base_params)
        keys = list(current_mean_params.keys())
        
        best_overall_code = parameterized_code
        best_overall_sharpe = -100.0
        
        for g in range(GENERATIONS):
            print(f"  >>> Generation {g+1}/{GENERATIONS} <<<")
            
            # Form population
            fitness_scores = np.zeros(POPULATION_SIZE)
            noise_matrix = np.random.randn(POPULATION_SIZE, len(keys))
            population_params = []
            
            for p in range(POPULATION_SIZE):
                cand_params = copy.deepcopy(current_mean_params)
                for k_idx, key in enumerate(keys):
                    val = cand_params[key]
                    if isinstance(val, int):
                        # mutate integers safely (e.g. windows)
                        mut = int(round(val + val * NOISE_STD * noise_matrix[p, k_idx]))
                        cand_params[key] = max(1, mut) # Never go below 1 for integers
                    elif isinstance(val, float):
                        mut = val + val * NOISE_STD * noise_matrix[p, k_idx]
                        cand_params[key] = mut
                population_params.append(cand_params)
                
                # Evaluate
                cand_code = self._inject_params_to_code(parameterized_code, cand_params)
                res = self.execute_evaluate(cand_code, f"Eggroll_G{g}_P{p}")
                
                if res['status'] == 'success':
                    fitness_scores[p] = res['sharpe']
                    if res['sharpe'] > best_overall_sharpe:
                        best_overall_sharpe = res['sharpe']
                        best_overall_code = cand_code
                        print(f"      [!] Local Peak Found: P{p} -> Sharpe {res['sharpe']:.4f}")
                else:
                    fitness_scores[p] = -100.0 # Heavy penalty for crashes
                    
            # Compute fitness weights (normalized rank or Z-score)
            f_mean = np.mean(fitness_scores)
            f_std = np.std(fitness_scores)
            if f_std < 1e-6: f_std = 1.0 # Prevent div zero
            
            weights = (fitness_scores - f_mean) / f_std
            
            # Step mean params towards successful perturbations
            print(f"      Gen {g+1} Mean Sharpe: {f_mean:.4f}")
            
            for k_idx, key in enumerate(keys):
                val = current_mean_params[key]
                step = LEARNING_RATE / (POPULATION_SIZE * NOISE_STD) * np.sum(weights * noise_matrix[:, k_idx])
                
                if isinstance(val, int):
                    new_val = int(round(val + val * step))
                    current_mean_params[key] = max(1, new_val)
                elif isinstance(val, float):
                    current_mean_params[key] = val + val * step
                    
            print(f"      Updated Mean Params: {current_mean_params}")
            
        print(f"[EGGROLL] Evolution Complete. Best Sharpe achieved: {best_overall_sharpe:.4f}")
        return best_overall_code, best_overall_sharpe
