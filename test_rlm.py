from src.core.rlm_scaffold import RLMScaffold

def fake_llm(messages, temperature=0, role_context=""):
    return "```python\nFinal = 'Hello World'\n```"

scaffold = RLMScaffold("Test", fake_llm)
res = scaffold.run_repl("Prompt")
print(res)
