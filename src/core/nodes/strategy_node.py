import re
import os


class StrategyNode:
    def __init__(self, harness_config, call_llm_func):
        self.config = harness_config
        self.call_llm = call_llm_func

    def _verify_integrity(self, knowledge_path):
        if not os.path.exists(knowledge_path):
            return True
        with open(knowledge_path, "r") as f:
            content = f.read()
            blocks = re.findall(r"### \[BLOCK_(\d+)\]", content)
            hashes = re.findall(r"(?<!Parent)Hash\*?\*?: ([a-f0-9]{64})", content)
            parents = re.findall(r"ParentHash\*?\*?: ([a-f0-9]{64})", content)

            if not blocks or not hashes or not parents:
                return True

            # Verify Chain only for closed blocks
            num_to_verify = min(len(blocks), len(hashes), len(parents))
            for i in range(1, num_to_verify):
                # Correct index?
                if int(blocks[i]) != i:
                    return False
                # Parent matches predecessor's hash?
                if parents[i] != hashes[i - 1]:
                    return False

            return True

    def execute(
        self,
        iteration,
        memory,
        knowledge,
        inspiration,
        failure_msg="",
        champion_info="",
    ):
        knowledge_file = "KNOWLEDGE.md"

        # Merkle Integrity Gate
        if not self._verify_integrity(knowledge_file):
            print("[SECURITY] !!! Merkle Chain Integrity Violation in KNOWLEDGE.md !!!")
            return None

        print("[NODE] > StrategyNode (Alpha/Beta Deliberation)...")

        # Agent Alpha
        alpha_info = self.config["agents"]["alpha"]
        prompt_alpha = alpha_info["template"].format(
            memory=memory,
            inspiration=inspiration,
            failure_msg=failure_msg,
            knowledge=knowledge,
        )
        # [V2.1 META-OPTIMIZATION]
        if champion_info:
            prompt_alpha += f"\n\n### CURRENT CHAMPION CONTEXT:\n{champion_info}\nFocus on refining this champion's parameters or identifying why its regime might be shifting."
        msgs_alpha = [
            {"role": "system", "content": alpha_info["system"]},
            {"role": "user", "content": prompt_alpha},
        ]
        pitch = self.call_llm(msgs_alpha, temperature=0.9, role_context="Alpha")

        if not pitch:
            return None

        # Agent Beta Critique
        beta_info = self.config["agents"]["beta_critic"]
        prompt_beta = beta_info["template"].format(pitch=pitch)
        msgs_beta = [
            {"role": "system", "content": beta_info["system"]},
            {"role": "user", "content": prompt_beta},
        ]
        critique = self.call_llm(msgs_beta, temperature=0.6, role_context="Beta")

        return {"pitch": pitch, "critique": critique}
