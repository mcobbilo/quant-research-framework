from src.core.rlm_scaffold import check_code_safety

malicious_code_1 = """
import os
os.system('echo Hacked')
"""

malicious_code_2 = """
().__class__.__bases__[0].__subclasses__()
"""

malicious_code_3 = """
exec('print("Hacked")')
"""

safe_code = """
import pandas as pd
"""  # Wait, my security blocks ALL imports. Is this safe? Yes, it will be blocked.

print("Test 1 (import):", check_code_safety(malicious_code_1))
print("Test 2 (dunder):", check_code_safety(malicious_code_2))
print("Test 3 (exec):", check_code_safety(malicious_code_3))
print("Test 4 (import pandas):", check_code_safety(safe_code))
