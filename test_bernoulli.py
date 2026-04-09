import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import xLSTM
from neuralforecast.losses.pytorch import DistributionLoss

try:
    df = pd.DataFrame({
        'unique_id': 'SPY',
        'ds': pd.date_range(start='2020-01-01', periods=100),
        'y': np.random.randint(0, 2, 100)
    })
    model = xLSTM(h=1, input_size=10, max_steps=5, loss=DistributionLoss(distribution='Bernoulli', level=[90]))
    nf = NeuralForecast(models=[model], freq='D')
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    nf.fit(df)
    res = nf.predict()
    print("OUTPUT FORMAT OF PREDICT:")
    print(res.columns)
    print(res.head())
except Exception as e:
    print(e)
