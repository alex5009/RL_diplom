import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
from collections import namedtuple
from scipy import stats

Prices = namedtuple('Prices', field_names=['open', 'close', 'volume', 'high', 'low', 'actions', 'rewards', 'time'])

DEFAULT_BARS_COUNT = 391
MAX_BAR_STEPS = 20
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Sell = 2


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, max_step_bars=MAX_BAR_STEPS, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC,
                 random_ofs_on_reset=False, volumes=True):
        assert isinstance(prices, dict)
        self._prices = prices
        if not random_ofs_on_reset:
            self.max_step_bars = DEFAULT_BARS_COUNT
        else:
            self.max_step_bars = max_step_bars
        self._state = State1D(bars_count, commission, volumes=volumes, max_step_bars=self.max_step_bars)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # в price можно поставить много акций
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument].copy()
        prices_copy = prices.copy()
        prices[['open', 'close', 'high', 'low', 'volume']] = prices[
            ['open', 'close', 'high', 'low', 'volume']].diff().fillna(0)
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(list(range(bars, prices.high.shape[0] - self.max_step_bars)))
        else:
            offset = bars
        self._state.reset(prices, offset, prices_copy, self.max_step_bars)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {'instrument': self._instrument, 'offset': self._state.offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]


class State1D:
    def __init__(self, bars_count, commission_perc, max_step_bars,
                 volumes=True):
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.volumes = volumes
        self.max_step_bars = max_step_bars
        self.done = False

    def reset(self, prices, offset, prices_copy, max_step_bars):
        self._prices = prices
        self.offset = offset
        self.reward = 0
        self.long_position = False
        self.short_position = False
        self.close_list = []
        self.prices_copy = prices_copy
        self.done = False
        self.max_step_bars = max_step_bars

    @property
    def shape(self):
        if self.volumes:
            return (8, self.bars_count)
        else:
            return (7, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count - 1
        dst = 0
        res[dst] = self._prices.high[self.offset - ofs:self.offset + 1]
        res[dst + 1] = self._prices.low[self.offset - ofs:self.offset + 1]
        res[dst + 2] = self._prices.close[self.offset - ofs:self.offset + 1]
        res[dst + 3] = self._prices.open[self.offset - ofs:self.offset + 1]
        if self.volumes:
            res[dst + 4] = self._prices.volume[self.offset - ofs:self.offset + 1] / 10000
            dst = 5
        else:
            dst = 4
        res[dst] = self._prices.time[self.offset - ofs:self.offset + 1]
        dst += 1
        res[dst] = self._prices.actions[self.offset - ofs:self.offset + 1]
        if self.long_position:
            res[dst][-1] = 1.
        elif self.short_position:
            res[dst][-1] = 2.
        else:
            res[dst][-1] = 0.
        self._prices.actions[self.offset - ofs:self.offset + 1] = res[dst]
        dst += 1
        res[dst] = self._prices.rewards[self.offset - ofs:self.offset + 1]
        res[dst][-1] = self.reward
        self._prices.rewards[self.offset - ofs:self.offset + 1] = res[dst]
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self.prices_copy.open[self.offset]
        rel_close = self.prices_copy.close[self.offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        reward = 0.0
        done = False
        close = self._cur_close()
        if not self.long_position and Actions.Buy == action:
            if not self.short_position:
                self.open_price = close
                reward -= self.commission_perc
                self.long_position = True

                # self.reward += -current_close * self.commission + (next_close - current_close)
                # self.close_list.append(current_close)
                # self.long_position = True
            else:
                reward -= self.commission_perc
                self.short_position = False
                self.open_price = 0.0

                # self.reward += -current_close * self.commission * len(self.close_list) + sum(np.array(self.close_list) - current_close)
                # self.close_list = []
                # self.short_position = False

        elif self.long_position and Actions.Buy == action:
            pass
            # self.reward += -next_close * self.commission * len(self.close_list) + sum(next_close - np.array(self.close_list))
            # self.reward += -current_close * self.commission + (next_close - current_close)
            # self.close_list.append(current_close)

        elif not self.short_position and Actions.Sell == action:
            if not self.long_position:
                self.open_price = close
                reward -= self.commission_perc * self.open_price
                self.short_position = True
                # self.reward += -current_close * self.commission + (current_close - next_close)
                # self.close_list.append(current_close)
                # self.short_position = True
            else:
                reward -= self.commission_perc * self.open_price
                self.long_position = False
                self.open_price = 0.0
                # self.reward += -current_close * self.commission * len(self.close_list) + sum(current_close - np.array(self.close_list))
                # self.close_list = []
                # self.long_position = False

        elif self.short_position and Actions.Sell == action:
            pass
            # self.reward += -next_close * self.commission * len(self.close_list) + sum(np.array(self.close_list) - next_close)
            # self.reward += -current_close * self.commission + (current_close - next_close)
            # self.close_list.append(current_close)

        # if action == Actions.Skip:
        #   if self.long_position:
        #     self.reward += -next_close * self.commission * len(self.close_list) + sum(next_close - np.array(self.close_list))
        #   elif self.short_position:
        #     self.reward += -next_close * self.commission * len(self.close_list) + sum(np.array(self.close_list) - next_close)
        self.offset += 1
        self.max_step_bars -= 1
        done |= self.max_step_bars <= 1
        prev_close = close
        close = self._cur_close()
        if self.long_position:
            reward += 100.0 * (close - prev_close) / prev_close
        elif self.short_position:
            reward += 100.0 * (prev_close - close) / prev_close
        self.reward = reward
        self.done = done
        return reward, done

        # if self.done and (self.long_position or self.short_position):
        #   if self.long_position:
        #     self.reward += -next_close * self.commission * len(self.close_list) + sum(next_close - np.array(self.close_list))
        #     self.long_position = False
        #     self.close_list = []
        #   elif self.short_position:
        #     self.reward += -next_close * self.commission * len(self.close_list) + sum(np.array(self.close_list) - next_close)
        #     self.short_position = False
        #     self.close_list = []
        # return self.reward, self.done

# import gym
# import gym.spaces
# from gym.utils import seeding
# import enum
# import numpy as np
# from collections import namedtuple
# import torch
# from scipy import stats
#
# Prices = namedtuple('Prices', field_names=['open', 'close', 'volume', 'high', 'low', 'actions', 'rewards', 'time'])
#
# DEFAULT_BARS_COUNT = 391
# MAX_BAR_STEPS = 20
# DEFAULT_COMMISSION_PERC = 0.1
#
#
# class Actions(enum.Enum):
#     Skip = 0
#     Buy = 1
#     Sell = 2
#
#
# class StocksEnv(gym.Env):
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self, prices, max_step_bars=MAX_BAR_STEPS, bars_count=DEFAULT_BARS_COUNT,
#                  commission=DEFAULT_COMMISSION_PERC,
#                  random_ofs_on_reset=False, volumes=True):
#         assert isinstance(prices, dict)
#         self._prices = prices
#         if not random_ofs_on_reset:
#             self.max_step_bars = DEFAULT_BARS_COUNT
#         else:
#             self.max_step_bars = max_step_bars
#         self._state = State1D(bars_count, commission, volumes=volumes, max_step_bars=self.max_step_bars)
#         self.action_space = gym.spaces.Discrete(n=len(Actions))
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
#         self.random_ofs_on_reset = random_ofs_on_reset
#         self.seed()
#
#     def reset(self):
#         # в price можно поставить много акций
#         self._instrument = self.np_random.choice(list(self._prices.keys()))
#         prices = self._prices[self._instrument].copy()
#         prices_copy = prices.copy()
#         prices[['open', 'close', 'high', 'low', 'volume']] = prices[
#             ['open', 'close', 'high', 'low', 'volume']].diff().fillna(0)
#         bars = self._state.bars_count
#         if self.random_ofs_on_reset:
#             offset = self.np_random.choice(list(range(bars, prices.high.shape[0] - self.max_step_bars)))
#         else:
#             offset = bars
#         self._state.reset(prices, offset, prices_copy, self.max_step_bars)
#         return self._state.encode()
#
#     def step(self, action_idx):
#         action = Actions(action_idx)
#         reward, done = self._state.step(action)
#         obs = self._state.encode()
#         info = {'instrument': self._instrument, 'offset': self._state.offset}
#         return obs, reward, done, info
#
#     def render(self, mode='human', close=False):
#         pass
#
#     def close(self):
#         pass
#
#     def seed(self, seed=None):
#         self.np_random, seed1 = seeding.np_random(seed)
#         seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
#         return [seed1, seed2]
#
#     # @classmethod
#     # def from_dir(cls, data_dir, **kwargs):
#     #     prices = {
#     #         file: load_relative(file) for file in price_files(data_dir)
#     #     }
#     #     return StockEnv(prices, **kwargs)
#
#
#
#
# class State1D:
#     def __init__(self, bars_count, commission_perc, max_step_bars,
#                  volumes=True):
#         self.bars_count = bars_count
#         self.commission = commission_perc
#         self.volumes = volumes
#         self.max_step_bars = max_step_bars
#         self.done = False
#
#     def reset(self, prices, offset, prices_copy, max_step_bars):
#         self._prices = prices
#         self.offset = offset
#         self.reward = 0
#         self.long_position = False
#         self.short_position = False
#         self.close_list = []
#         self.prices_copy = prices_copy
#         self.done = False
#         self.max_step_bars = max_step_bars
#
#     @property
#     def shape(self):
#         if self.volumes:
#             return (8, self.bars_count)
#         else:
#             return (7, self.bars_count)
#
#     def encode(self):
#         res = np.zeros(shape=self.shape, dtype=np.float32)
#         ofs = self.bars_count - 1
#         dst = 0
#         res[dst] = self._prices.high[self.offset - ofs:self.offset + 1]
#         res[dst + 1] = self._prices.low[self.offset - ofs:self.offset + 1]
#         res[dst + 2] = self._prices.close[self.offset - ofs:self.offset + 1]
#         res[dst + 3] = self._prices.open[self.offset - ofs:self.offset + 1]
#         if self.volumes:
#             res[dst + 4] = self._prices.volume[self.offset - ofs:self.offset + 1] / 10000
#             dst = 5
#         else:
#             dst = 4
#         res[dst] = self._prices.time[self.offset - ofs:self.offset + 1]
#         dst += 1
#         res[dst] = self._prices.actions[self.offset - ofs:self.offset + 1]
#         if self.long_position:
#             res[dst][-1] = 1.
#         elif self.short_position:
#             res[dst][-1] = 2.
#         else:
#             res[dst][-1] = 0.
#         self._prices.actions[self.offset - ofs:self.offset + 1] = res[dst]
#         dst += 1
#         res[dst] = self._prices.rewards[self.offset - ofs:self.offset + 1]
#         res[dst][-1] = self.reward
#         self._prices.rewards[self.offset - ofs:self.offset + 1] = res[dst]
#         return res
#
#     def step(self, action):
#         # done = False
#         current_close = self.prices_copy.close[self.offset]
#         next_close = self.prices_copy.close[self.offset + 1]
#         if not self.long_position and Actions.Buy == action:
#             if not self.short_position:
#                 self.reward += -current_close * self.commission + (next_close - current_close)
#                 self.close_list.append(current_close)
#                 self.long_position = True
#             else:
#                 self.reward += -current_close * self.commission * len(self.close_list) + sum(
#                     np.array(self.close_list) - current_close)
#                 self.close_list = []
#                 self.short_position = False
#
#         elif self.long_position and Actions.Buy == action:
#             self.reward += -current_close * self.commission + (next_close - current_close)
#             self.close_list.append(current_close)
#
#         elif not self.short_position and Actions.Sell == action:
#             if not self.long_position:
#                 self.reward += -current_close * self.commission + (current_close - next_close)
#                 self.close_list.append(current_close)
#                 self.short_position = True
#             else:
#                 self.reward += -current_close * self.commission * len(self.close_list) + sum(
#                     current_close - np.array(self.close_list))
#                 self.close_list = []
#                 self.long_position = False
#
#         elif self.short_position and Actions.Sell == action:
#             self.reward += -current_close * self.commission + (current_close - next_close)
#             self.close_list.append(current_close)
#
#         if action == Actions.Skip:
#             if self.long_position:
#                 self.reward += -next_close * self.commission * len(self.close_list) + sum(
#                     next_close - np.array(self.close_list))
#             elif self.short_position:
#                 self.reward += -next_close * self.commission * len(self.close_list) + sum(
#                     np.array(self.close_list) - next_close)
#         self.offset += 1
#         self.max_step_bars -= 1
#         self.done |= self.max_step_bars <= 1
#
#         if self.done and (self.long_position or self.short_position):
#             if self.long_position:
#                 self.reward += -next_close * self.commission * len(self.close_list) + sum(
#                     next_close - np.array(self.close_list))
#                 self.long_position = False
#                 self.close_list = []
#             elif self.short_position:
#                 self.reward += -next_close * self.commission * len(self.close_list) + sum(
#                     np.array(self.close_list) - next_close)
#                 self.short_position = False
#                 self.close_list = []
#         return self.reward, self.done
#
#
# def calculate_reward(env_val, net, device):
#     val_share = env_val.reset()
#     res_reward = []
#     acts = []
#     for _ in range(0, 391):
#         x = torch.tensor(val_share.reshape(1, 8, -1)).to(device)
#         logits = np.asarray(net(x).detach().cpu().numpy())
#         act = logits.argmax()
#         val_share, rew, done, inst = env_val.step(act)
#         res_reward.append(rew)
#         acts.append(act)
#         if done:
#             break
#     return res_reward[-1], np.array(acts).std()



# import gym
# import gym.spaces
# from gym.utils import seeding
# import enum
# import numpy as np
# from collections import namedtuple
#
# Prices = namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])
#
# DEFAULT_BARS_COUNT = 391
# DEFAULT_COMMISSION_PERC = 0.1
#
#
# class Actions(enum.Enum):
#     Skip = 0
#     Buy = 1
#     Sell = 2
#
#
# class StocksEnv(gym.Env):
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT, commission=DEFAULT_COMMISSION_PERC,
#                  reset_on_close=True, state_1d=False, random_ofs_on_reset=False, reward_on_close=False, volumes=True):
#         assert isinstance(prices, dict)
#         self._prices = prices
#         if not state_1d:
#             self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
#                                 volumes=volumes)
#         else:
#             self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
#                                   volumes=volumes)
#         self.action_space = gym.spaces.Discrete(n=len(Actions))
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
#         self.random_ofs_on_reset = random_ofs_on_reset
#         self.seed()
#
#     def reset(self):
#         # в price можно поставить много акций
#         self._instrument = self.np_random.choice(list(self._prices.keys()))
#         prices = self._prices[self._instrument]
#         bars = self._state.bars_count
#         if self.random_ofs_on_reset:
#             offset = self.np_random.choice(prices.high.shape[0] - bars * 10) + bars
#         else:
#             offset = bars
#         self._state.reset(prices, offset)
#         return self._state.encode()
#
#     def step(self, action_idx):
#         action = Actions(action_idx)
#         reward, done = self._state.step(action)
#         obs = self._state.encode()
#         info = {'instrument': self._instrument, 'offset': self._state.offset}
#         return obs, reward, done, info
#
#     def render(self, mode='human', close=False):
#         pass
#
#     def close(self):
#         pass
#
#     def seed(self, seed=None):
#         self.np_random, seed1 = seeding.np_random(seed)
#         seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
#         return [seed1, seed2]
#
#     @classmethod
#     def from_dir(cls, data_dir, **kwargs):
#         prices = {
#             file: load_relative(file) for file in price_files(data_dir)
#         }
#         return StockEnv(prices, **kwargs)
#
#
# class State:
#     def __init__(self, bars_count, commission_perc, reset_on_close,
#                  reward_on_close=True, volumes=True):
#         # assert isinstance(bars_count, int)
#         # assert bars_count > 0
#         assert isinstance(commission_perc, float)
#         assert commission_perc >= 0.
#         assert isinstance(reset_on_close, bool)
#         assert isinstance(reward_on_close, bool)
#         self.bars_count = bars_count
#         self.commission_perc = commission_perc
#         self.reset_on_close = reset_on_close
#         self.reward_on_close = reward_on_close
#         self.volumes = volumes
#
#     def reset(self, prices, offset):
#         # assert isinstance(prices, Prices)
#         assert offset >= self.bars_count - 1
#         self.have_position = False
#         self.open_price = 0.
#         self._prices = prices
#         self.offset = offset
#         self.reward_long = 0.0
#         self.reward_short = 0.0
#         self.long_position = None
#         self.short_position = None
#         self.amount_stocks_buy = 0
#         self.amount_stocks_sell = 0
#
#     @property
#     def shape(self):
#         if self.volumes:
#             return (4 * self.bars_count + 1 + 1,)
#         else:
#             return (3 * self.bars_count + 1 + 1,)
#
#     def encode(self):
#         res = np.ndarray(shape=self.shape, dtype=np.float32)
#         shift = 0
#         for bar_idx in range(-self.bars_count + 1, 1):
#             res[shift] = self._prices.high[self.offset + bar_idx]
#             shift += 1
#             res[shift] = self._prices.low[self.offset + bar_idx]
#             shift += 1
#             res[shift] = self._prices.close[self.offset + bar_idx]
#             shift += 1
#             if self.volumes:
#                 res[shift] = self._prices.volume[self.offset + bar_idx] / 10000
#                 shift += 1
#         res[shift] = float(self.have_position)
#         shift += 1
#         if not self.have_position:
#             res[shift] = 0.
#         else:
#             res[shift] = (self._cur_close() - self.open_price) / self.open_price
#         return res
#
#     def _cur_close(self):
#         open = self._prices.open[self.offset]
#         rel_close = self._prices.close[self.offset]
#         return open * (1. + rel_close)
#
#     def step(self, action):
#         done = False
#         close = self._cur_close()
#         reward = 0.
#         if self.long_position is None and self.short_position is None and action != Actions.Skip:
#             #  открытие позиции в шорт или лонг
#             if action == Actions.Buy:
#                 self.amount_stocks_buy += 1
#                 self.reward_long += -close * self.commission_perc
#                 self.long_position = True
#
#             elif action == Actions.Sell:
#                 self.amount_stocks_sell += 1
#                 self.reward_short += -self.commission_perc * close
#                 self.short_position = True
#             self.open_price = close
#         # обработка открытой лонг позиции
#         elif self.long_position and action != Actions.Skip:
#             if action == Actions.Buy:
#                 # pass
#                 self.amount_stocks_buy += 1
#                 self.reward_long += -close * self.commission_perc
#
#             elif action == Actions.Sell:
#                 self.reward_long += (close - self.open_price) * self.amount_stocks_buy - self.commission_perc * close
#                 self.amount_stocks_buy = 0
#                 self.long_position = None
#
#         # обработка открытой шорт позиции
#         elif self.short_position and action != Actions.Skip:
#             if action == Actions.Buy:
#                 self.reward_short += (self.open_price - close) * self.amount_stocks_sell - self.commission_perc * close
#                 self.amount_stocks_sell = 0
#                 self.short_position = None
#                 # reward = self.reward_short + self.reward_long
#             elif action == Actions.Sell:
#                 # pass
#                 self.amount_stocks_sell += 1
#                 self.reward_short += (-self.commission_perc * close)
#
#         reward = self.reward_short + self.reward_long
#         self.offset += 1
#         prev_close = close
#         close = self._cur_close()
#         done |= self.offset >= self._prices.close.shape[0] - 1
#
#         if done and (self.long_position or self.short_position):
#             if self.long_position:
#                 self.reward_long += (close - self.open_price) * self.amount_stocks_buy - self.commission_perc * close
#                 self.amount_stocks_buy = 0
#                 self.long_position = None
#             else:
#                 self.reward_short += (self.open_price - close) * self.amount_stocks_buy - self.commission_perc * close
#                 self.amount_stocks_sell = 0
#                 self.short_position = None
#             reward = self.reward_short + self.reward_long
#         elif (self.long_position or self.short_position) and action == Actions.Skip:
#             if self.long_position:
#                 reward = (close - self.open_price) * self.amount_stocks_buy - self.commission_perc * close
#             elif self.short_position:
#                 reward = (self.open_price - close) * self.amount_stocks_buy - self.commission_perc * close
#         elif not (self.long_position or self.short_position) and action == Actions.Skip:
#             reward = 0
#         if done:
#             self.reward_long = 0.0
#             self.reward_short = 0.0
#         # print('short', self.reward_short, '\nlong', self.reward_long, '\nreward', reward, '\nclose', close, '\nprev_close', prev_close)
#         return reward, done
#
#
# class State1D(State):
#     @property
#     def shape(self):
#         if self.volumes:
#             return (6, self.bars_count)
#         else:
#             return (5, self.bars_count)
#
#     def encode(self):
#         res = np.zeros(shape=self.shape, dtype=np.float32)
#         ofs = self.bars_count - 1
#         res[0] = self._prices.high[self.offset - ofs:self.offset + 1]
#         res[1] = self._prices.low[self.offset - ofs:self.offset + 1]
#         res[2] = self._prices.close[self.offset - ofs:self.offset + 1]
#         if self.volumes:
#             res[3] = self._prices.volume[self.offset - ofs:self.offset + 1]
#             dst = 4
#         else:
#             dst = 3
#         if self.have_position:
#             res[dst] = 1.
#             res[dst + 1] = (self._cur_close() - self.open_price) / self.open_price
#         return res