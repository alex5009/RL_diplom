import ptan
import json
import torch.optim as optim
from gym import wrappers

from load_data import *
from environ import StocksEnv
from model import DQNConv1D, DQNConv1DLarge
from common import *
import warnings
from validation import *
from monte_carlo import *
warnings.filterwarnings('ignore')



BATCH_SIZE = 128
BARS_COUNT = 391
TARGET_NET_SYNC = 1000
GAMMA = 0.95

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 4

LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1
EPSILON_STOP = 0.05
EPSILON_STEPS = 700000

CHECKPOINT_EVERY_STEP = 10000
VALIDATION_EVERY_STEP = 1000
EVAL_EVERY_STEP_VAL = EVAL_EVERY_STEP * 10
best_step_val = -np.inf

device = 'cuda:0'
cols = ['open', 'close', 'volume', 'high', 'low', 'actions', 'rewards', 'time']

if __name__ == '__main__':
    loader = GetDataPnD()
    loader.load_from_db()
    stock_data = loader._prices
    train_data = {}
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
        if len(train_data) > 2000:
            break
    stock_data.update(train_data)
    val_data = loader._prices_val

    env = StocksEnv(stock_data, bars_count=BARS_COUNT, volumes=True, random_ofs_on_reset=False, max_step_bars=30)

    env = wrappers.TimeLimit(env, max_episode_steps=1000)
    env_val = StocksEnv(val_data, bars_count=BARS_COUNT, volumes=True)

    net = DQNConv1DLarge(env.observation_space.shape, env.action_space.n).to(device)
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    eval_states = None
    best_mean_val = None
    all_loss_v = []
    while True:
        step_idx += 1
        buffer.populate(1)
        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

        new_rewards = exp_source.pop_rewards_steps()

        if len(buffer) < REPLAY_INITIAL:
            continue

        if eval_states is None:
            print('Initial buffer populated, start training')
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
            eval_states = np.array(eval_states, copy=False)

        if step_idx % EVAL_EVERY_STEP == 0:
            mean_val = calc_values_of_states(eval_states, net, device=device)
            if best_mean_val is None or best_mean_val < mean_val:
                if best_mean_val is not None:
                    print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                best_mean_val = mean_val

        if step_idx % EVAL_EVERY_STEP_VAL == 0:
            mean_val1, std_actions1 = calculate_reward(env_val, net, device)
            mean_val2, std_actions2 = calculate_reward(env_val, net, device)
            mean_val3, std_actions3 = calculate_reward(env_val, net, device)
            mean_val4, std_actions4 = calculate_reward(env_val, net, device)
            mean_step_val = np.array([mean_val1, mean_val2, mean_val3, mean_val4]).mean()
            mean_std_act = np.array([std_actions1, std_actions2, std_actions3, std_actions4]).mean()
            print(f'Validation steps: {step_idx} | mean_val: {mean_step_val} | std actions {mean_std_act}')
            if mean_step_val > best_step_val:
                best_step_val = mean_step_val
                torch.save(net.state_dict(), 'result_model.pth')
                print(f'Save model step {step_idx}')

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_v = calc_loss(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
        loss_v.backward()
        optimizer.step()

        if step_idx % TARGET_NET_SYNC == 0:
            tgt_net.sync()
        if step_idx % 1000 == 0:
            print(f'Steps: {step_idx} | mean_val: {mean_val} | loss: {loss_v}| epsilon {selector.epsilon}')

