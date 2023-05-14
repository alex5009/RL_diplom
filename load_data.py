import numpy as np
from collections import namedtuple
import alpaca_trade_api as tradeapi
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta
import time
import glob
from connect_data import *


class GetDataPnD():
    def __init__(self):
        self._prices = {}
        self._prices_val = {}
        self.counter = 0
        self.connect()

    def connect(self):
        
        alpaca_endpoint = 'https://paper-api.alpaca.markets'
        self.api = tradeapi.REST(key, secret, alpaca_endpoint)

    def db_connect(self):
        

        self.engine = create_engine(f"mysql://{user}:{password}@{host}/alpaca")

    def read_data(self, df, filter_data=True, fix_open_price=False):
        data = df.to_numpy().T
        prices = namedtuple('prices', field_names=['open', 'high', 'low', 'close', 'volume'])
        return prices(open=np.array(data[0], dtype=np.float32),
                      high=np.array(data[1], dtype=np.float32),
                      low=np.array(data[2], dtype=np.float32),
                      close=np.array(data[3], dtype=np.float32),
                      volume=np.array(data[4], dtype=np.float32))

    def prices_to_relative(self, prices):
        rh = (prices.high - prices.open) / prices.open
        rl = (prices.low - prices.open) / prices.open
        rc = (prices.close - prices.open) / prices.open
        prices1 = namedtuple('prices', field_names=['open', 'high', 'low', 'close', 'volume'])
        return prices1(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)

    def load_relative(self, data):
        return self.prices_to_relative(self.read_data(data))

    def compare_price(self, x):
        high_price = x.pct_change().tolist()
        high_price.reverse()
        negative_day = [i for i, v in enumerate(high_price) if v > 0]
        default_delta = 3
        if len(negative_day) > 0:
            delta = default_delta if negative_day[0] < default_delta else negative_day[0]
        else:
            delta = default_delta
        prices = x.to_numpy()
        last = prices[-1]

        diff_prices = x.iloc[-1] - x.iloc[-delta]
        ratio = round(last / prices[-delta], 2)
        return ((ratio > 2) or (diff_prices > 2 and ratio > 1.37) or (diff_prices > 4 and ratio > 1.2)) and (ratio < 10)

    def time_market(self, dt):
        min_time = dt + pd.Timedelta(9.5, unit='hour')
        max_time = dt + pd.Timedelta(16, unit='hour')
        idx_ = pd.date_range(min_time, max_time, freq='1min')
        return idx_

    def walk_to_series(self, x):
        x = x.dropna()
        dates = []
        delta = 10 
        end_idx = 10  
        for i in range(len(x) - delta):
            if self.compare_price(x.iloc[end_idx - delta:end_idx]):
                dates.append(x.index[end_idx])
            end_idx += 1
        return dates

    def add_in_base(self, d):
        sym = d.symbol
        for date_ in d.dates:
            end_date_old = date_
            wd = date_.weekday()
            if wd == 0:
                # если выпадает на понедельник, то берём данные с пятницы
                start_date = (date_ - pd.Timedelta(3, unit='day'))
            else:
                start_date = (date_ - pd.Timedelta(1, unit='day'))
            end_date = (date_ + pd.Timedelta(1, unit='day'))
            df_alpaca = self.api.get_barset(sym, 'minute', start=start_date.isoformat(), end=end_date.isoformat(),
                                            limit=1000).df
            if df_alpaca.shape[0] > 300 and df_alpaca.loc[end_date_old:].shape[0] > 180:
                dates_range = self.time_market(start_date).union(self.time_market(end_date_old))
                df_alpaca = df_alpaca.reindex(dates_range, method='bfill').replace(0, method='bfill').fillna(
                    method='ffill')
                self.counter += 1
                if sym not in self._prices.keys():
                    self._prices.update({sym: []})
                self._prices[sym].append(self.load_relative(df_alpaca))

    def add_in_aws(self, d):
        sym = d.symbol
        for date_ in d.dates:
            end_date_old = date_
            wd = date_.weekday()
            if wd == 0:
                # если выпадает на понедельник, то берём данные с пятницы
                start_date = (date_ - pd.Timedelta(3, unit='day'))
            else:
                start_date = (date_ - pd.Timedelta(1, unit='day'))
            end_date = (date_ + pd.Timedelta(1, unit='day'))
            df_alpaca = self.api.get_barset(sym, 'minute', start=start_date.isoformat(), end=end_date.isoformat(),
                                            limit=1000).df
            if df_alpaca.shape[0] > 300 and df_alpaca.loc[end_date_old:].shape[0] > 180:
                dates_range = self.time_market(start_date).union(self.time_market(end_date_old))
                df_alpaca = df_alpaca.reindex(dates_range, method='bfill').replace(0, method='bfill').fillna(
                    method='ffill')
                target_day = ''.join((date_ - timedelta(days=1)).isoformat().split('-'))
                table_name = target_day + '_' + sym
                try:
                    df_alpaca.to_sql(con=self.engine, name=table_name, if_exists='fail', index=False)
                except ValueError as ve:
                    print(ve)

    def search_PnD(self, list_symbols, save_in_db=False):
        count_sym = 200
        max_len = int(len(list_symbols) / count_sym) + 1
        df_pnd_full = pd.DataFrame()
        for en in range(max_len):
            df = self.api.get_barset(list_symbols[en * count_sym:(en + 1) * count_sym], 'day', limit=1000).df
            df_list_pnd = df.loc[:, [t for t in df.columns if t[1] == 'high']].apply(self.walk_to_series, axis=0)
            df_pnd = df_list_pnd[df_list_pnd.apply(lambda a: len(a) > 0)].reset_index(level=1, drop=True).reset_index()
            df_pnd.columns = ['symbol', 'dates']
            #
            if save_in_db:
                self.db_connect()
                df_pnd.apply(self.add_in_aws, axis=1)
            else:
                df_pnd.apply(self.add_in_base, axis=1)

    def load_from_db(self):
        self.db_connect()
        q = self.engine.execute('SHOW TABLES')
        available_tables = q.fetchall()
        train_tickers = available_tables[:-1200]
        val_tickers = available_tables[-1200:]
        for tab in train_tickers:
            query = f'SELECT * FROM {tab[0]}'
            df_alpaca = pd.read_sql(query, con=self.engine).iloc[:, 1:]
            df_alpaca.columns = [i.split(', ')[-1].split("'")[1] for i in df_alpaca.columns]
            self._prices.update({tab[0]: []})
            self._prices[tab[0]].append(self.load_relative(df_alpaca))

        for tab in val_tickers:
            query = f'SELECT * FROM {tab[0]}'
            df_alpaca = pd.read_sql(query, con=self.engine).iloc[:, 1:]
            df_alpaca.columns = [i.split(', ')[-1].split("'")[1] for i in df_alpaca.columns]
            self._prices_val.update({tab[0]: []})
            self._prices_val[tab[0]].append(self.load_relative(df_alpaca))

if __name__ == '__main__':
    data_dict = {}
    for i in glob.glob('data/*'):
        name = i.split('/')[-1]
        data = pd.read_csv(i)
        data_dict.update({name: data})
