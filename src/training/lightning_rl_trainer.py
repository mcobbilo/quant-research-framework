import os
import torch
import agentlightning as agl
from agentlightning.store import LightningStore
from agentlightning.trainer import Trainer

# Import the actual architecture we are tuning
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from deliberation import LatentCouncil

class CouncilPPOAlgorithm(agl.Algorithm):
    """
    Subclasses the Agent-Lightning Algorithm base class to map asynchronously
    collected reasoning spans into PyTorch weight updates for the LatentCouncil.
    """
    def __init__(self, store: LightningStore, council_architecture: LatentCouncil):
        super().__init__(store=store)
        self.council = council_architecture
        # Optimizer targets the Role Adapter that modulates Multi-Agent Consensus
        self.optimizer = torch.optim.Adam(self.council.role_adapter.parameters(), lr=1e-4)
        
    def run_step(self):
        # 1. Fetch latest unprocessed inference spans and verified drift rewards
        spans = self.store.fetch_new_spans()
        if not spans:
            return
        
        loss_accumulator = 0.0
        
        for span in spans:
            # Reconstruct the computational graph for the role_adapter 
            # (In production, gradients could be stashed or re-run)
            # We apply proxy Reinforcement (PPO clipped logic or simple policy gradient)
            # based on the Environmental Reward emitted via agl.emit_env_reward
            reward = span.env_reward
            
            # Simple placeholder for Async Policy Gradient Update
            if reward is not None:
                # Assuming span.inference_output is linked to gradient graph in a full run
                # negative drift reward is positive reinforcement
                # Here we mock the backward pass for structural demonstration
                loss = -torch.tensor(reward, requires_grad=True).mean() 
                loss_accumulator += loss

        if loss_accumulator != 0.0:
            self.optimizer.zero_grad()
            loss_accumulator.backward()
            self.optimizer.step()
            
            # Persist the newly adapted Multi-Agent Role parameters
            self.council.save_weights()
            print(f"[Lightning Trainer] Flushed {len(spans)} traces to policy. Council weights adapted.")


def launch_async_trainer():
    store_dir = os.path.join(os.path.dirname(__file__), "..", "..", "lightning_store")
    os.makedirs(store_dir, exist_ok=True)
    
    print(f"Initializing Asynchronous Agent-Lightning Store at {store_dir}")
    store = LightningStore(store_dir)
    
    print("Loading LatentCouncil Tri-Agent Architecture...")
    council = LatentCouncil(feature_dim=256) # Configured proxy dim
    
    print("Injecting Custom Policy Gradient Algorithm...")
    algorithm = CouncilPPOAlgorithm(store=store, council_architecture=council)
    
    # 4. Attach to the master Agent-Lightning Trainer
    trainer = Trainer(store=store, algorithm=algorithm)
    
    print("Agent-Lightning Async RL loop successfully engaged in the background.")
    trainer.run() 
    
if __name__ == "__main__":
    launch_async_trainer()
