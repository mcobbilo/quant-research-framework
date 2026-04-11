import pandas.util._decorators as decorators


def mocked_deprecate_kwarg(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


# Patch the pandas 3.0 deprecate_kwarg method that breaks pandas-datareader
decorators.deprecate_kwarg = mocked_deprecate_kwarg

import pandas_datareader.data as web
import datetime

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2024, 1, 1)

try:
    print("Testing ALFRED data pull for RECPROUSM156N...")
    # 'alfred' fetches the first release vintage data to avoid revisions
    df = web.DataReader("RECPROUSM156N", "alfred", start, end)
    print("SUCCESS: Retrieved ALFRED Recession Probability")
    print(df.tail(3))
except Exception as e:
    print(f"FAILED ALFRED RECPRO: {e}")
