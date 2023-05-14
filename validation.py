import numpy as np
import torch
from environ import Actions


def validation_run(env, net, episodes=100, device='cpu', epsilon=0.02, comission=0.1):
  stats = {
      'episode_reward': [],
      'episode_steps': [],
      'order_profits': [],
      'order_steps': []
  }
  for episode in range(episodes):
    obs = env.reset()
    total_reward=0.0
    position = None
    position_steps = None
    episode_steps = 0
    while True:
      obs_v = torch.tensor([obs]).to(device)
      out_v = net(obs_v)

      action_idx = out_v.max(dim=1)[1].item()
      if np.random.random() < epsilon:
        action_idx = env.action_space.sample()
      action = Actions(action_idx)

      close_price = env._state._cur_close()

      if action == Actions.Buy and position is None:
        position = close_price
        position_steps = 0
      elif action == Actions.Sell and position is not None:
        profit = close_price - position - (close_price + position) * comission / 100
        profit = 100 * profit / position
        stats['order_profits'].append(profit)
        stats['order_steps'].append(position_steps)
        position = None
        position_steps = None


      obs, reward, done, _ = env.step(action_idx)
      total_reward += reward
      episode_steps += 1
      if position_steps is not None:
        position_steps += 1
      if done:
        if position is not None:
          profit = close_price - position - (close_price + position) * comission / 100
          profit = 100 * profit / position
          stats['order_profits'].append(profit)
          stats['order_steps'].append(position_steps)
        break

    stats['episode_reward'].append(total_reward)
    stats['episode_steps'].append(episode_steps)

  return {key: np.mean(vals) for key, vals in stats.items()}

class Profit:
  def __init__(self, df, commission=0.1):
    self.df = df
    self.commission = commission
    self.actual_action = 0
    self.profit = 0
    self.counter = 0
    self.open_price = None

  def action(self, act, done):
    self.counter += 1
    if not done:
      if act == 0 or act == self.actual_action:
        pass
      elif act == 1 and self.actual_action != 1:
        if self.actual_action == 0:
          self.open_price = self.df['close'].iloc[self.counter]
          self.actual_action = 1
        else:
          self.actual_action = 0
          close = self.df['close'].iloc[self.counter]
          self.profit += -self.commission * self.open_price + (close - self.open_price)
          self.open_price = None

      elif act == 2 and self.actual_action != 2:
        if self.actual_action == 0:
          self.open_price = self.df['close'].iloc[self.counter]
          self.actual_action = 2
        else:
          self.actual_action = 0
          close = self.df['close'].iloc[self.counter]
          self.profit += -self.commission * self.open_price + (self.open_price - close)
          self.open_price = None

def calculate_reward(env_val, net, device):
  with torch.no_grad():
    val_share = env_val.reset()
    df = env_val._state.prices_copy.iloc[391:][['open', 'close']]
    profit = Profit(df)
    acts = []
    for _ in range(0,391):
      x = torch.tensor(val_share.reshape(1, 8, -1)).to(device)
      logits = np.asarray(net(x)[0].detach().cpu().numpy())
      act = logits.argmax()
      val_share, rew, done, inst = env_val.step(act)
      profit.action(act, done)
      acts.append(act)
      if done:
        break
  return profit.profit, np.array(acts).std()