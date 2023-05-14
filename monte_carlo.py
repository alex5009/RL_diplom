import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool
from scipy import stats
import time

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.api import VAR

train_data = {}
np.random.seed(0)
first_lag = 394
eps = 0.1


class predict_volume:
  def __init__(self, data, lags, shift_=2):
    self.data = data
    self.lags = lags
    self.shift_ = shift_
    self.shape0 = data.shape[0]

  def fit(self):
    ts = self.data[['open', 'volume']].copy()
    ts['open'] = ts['open'].shift(self.shift_)
    d = ts.dropna().diff().dropna()
    self.model = VAR(d).fit(self.lags)

  def generate_volume(self, generated_data, first_index):
    np.random.seed(0)
    new_data = generated_data[['open', 'volume']].to_numpy()[:first_index]
    for i in range(first_index, self.shape0):
      X = new_data.copy()
      last_value = X[-1, 1]
      X[:, 0] = shift(X[:, 0], self.shift_, cval=np.NaN)
      input_x = np.diff(X[self.shift_:], axis=0)[-self.lags:]
      y_forecast = self.model.forecast(input_x, 1)[0][1] + last_value
      if y_forecast <= 0:
        y_forecast = np.random.exponential(0.5)
      new_data = np.concatenate((new_data , np.array([[generated_data['open'].iloc[i], y_forecast]])))
    return new_data

def simulate_data(x):
    d = x.diff().fillna(0)
    first_value = x[first_lag]
    mu, exponential_decay = stats.laplace.fit(d[first_lag:])
    open = np.random.laplace(mu, exponential_decay, (782-first_lag, 2000))
    max_value = x[391:].max() + x[391:].max() * eps
    tail_max_value = x[600:].max()
    real_open = (open.cumsum(axis=0) + first_value)
    mask = (((real_open.max(axis=0) <= max_value).astype(int) + 
             (real_open[200:].max(axis=0) <= tail_max_value).astype(int)) / 2).round().astype(bool)
    if mask.sum() == 0:
      mask[0] = True
    open = open[:, mask]
    head = np.tile(d[:first_lag].cumsum() + x[0], (mask.sum(), 1)).T
    tail = open.cumsum(axis=0) + first_value
    res = np.concatenate([head, tail])
    return res

def close_simulate(data):
  d = (data['open'] - data['close'])
  mu, exponential_decay = stats.laplace.fit(d[first_lag:])
  diff = np.random.laplace(mu, exponential_decay, size=782-first_lag)
  res = np.concatenate([d[:first_lag], diff])
  return res

def func_generate(i):
    d = pd.DataFrame()
    d['open'] = open[:, i]
    d['close'] = close[:, i]
    d['volume'] = prices['volume']
    res_volume = volume.generate_volume(d, first_lag)[:,1]
    d['volume'] = res_volume / 10000
    d['high'] = d[['open', 'close']].max(axis=1) + np.random.exponential(0.13)
    d['low'] = d[['open', 'close']].min(axis=1) - np.random.exponential(0.13)
    d['actions'] = 0
    d['rewards'] = 0
    d['time'] = d.index
    return {name+f'_{i}': d}

if __name__ == '__main__':
  for name, prices in stock_data.items():
    start = time.time()
    open = simulate_data(prices['open'])
    close_diff = close_simulate(prices)
    close = (open - np.tile(close_diff, (open.shape[1], 1)).T)
    volume = predict_volume(prices, 100)
    volume.fit()

    with Pool(20) as p:
        res = p.map(func_generate, list(range(close.shape[1])))
    for val in res:
      train_data.update(val)
    print(name, time.time() - start)
    if len(train_data) > 20000:
      break