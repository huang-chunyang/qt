import numpy as np
import argparse
from copy import deepcopy
import random
import torch
from timeit import default_timer as timer
from models import Net
from tianshou.env import DummyVectorEnv
from custom_environment import FuturesTradingEnv

from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import offpolicy_trainer
from torch.optim import Adam

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Financial Forecasting -- deep reinforcement learning + RNN + MLP')

    ## 1. Actor setting
    parser.add_argument('--init_w', default=0.05, type=float, help='initialize model weights') 
    parser.add_argument('--hidden_fc1', default=256, type=int, help='hidden num of 1st-fc layer')
    parser.add_argument('--hidden_fc2', default=64, type=int, help='hidden num of 2nd-fc layer')
    parser.add_argument('--hidden_fc3', default=32, type=int, help='hidden num of 3rd-fc layer')

    ## 2. RNN setting
    parser.add_argument('--input_size', default=5, type=int, help='num of features for input state')
    parser.add_argument('--seq_len', default=15, type=int, help='sequence length of input state')
    parser.add_argument('--num_rnn_layer', default=2, type=int, help='num of rnn layer')
    parser.add_argument('--hidden_rnn', default=128, type=int, help='hidden num of lstm layer')

    ## 3. Learning setting
    parser.add_argument('--r_rate', default=0.0001, type=float, help='gru layer learning rate')
    parser.add_argument('--beta1', default=0.3, type=float, help='mometum beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.9, type=float, help='mometum beta2 for Adam optimizer')
    parser.add_argument('--sch_step_size', default=16*150, type=float, help='LR_scheduler: step_size')
    parser.add_argument('--sch_gamma', default=0.5, type=float, help='LR_scheduler: gamma')
    parser.add_argument('--bsize', default=1, type=int, help='minibatch size')
    ## 4. misc 
    parser.add_argument('--ou_theta', default=0.18, type=float, help='noise theta of Ornstein Uhlenbeck Process')
    parser.add_argument('--ou_sigma', default=0.3, type=float, help='noise sigma of Ornstein Uhlenbeck Process') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu of Ornstein Uhlenbeck Process') 
    args = parser.parse_args()

    ## prepare agent model
    ### Net 有两部分组成： RNN 用来分析传入的序列的特征， Actor 用来产生行为输出。
    net = Net(args)
    optim = Adam(net.parameters(), lr=1e-3)

    ## parare drl env 
    env = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC", data_mode="train") for _ in range(2)])
    train_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC", data_mode="train") for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC", data_mode="test") for _ in range(1)])

    action_space = env.action_space[0]
    action_shape = action_space.shape or action_space.n
    state_shape = env.observation_space[0]
    print("The shape of action is", action_shape, " The shape of state is ", state_shape)
    policy = DQNPolicy(net, optim,discount_factor=0.95,
                        estimation_step=2, target_update_freq=500)
    policy.set_eps(0.5)

    ### prepare data collector
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=50000, buffer_num=args.bsize))
    test_collector = Collector(policy, test_envs)

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=25, step_per_epoch=2400, step_per_collect=50,
        update_per_step=0.1, episode_per_test=10, batch_size=64)
    
    for one in result:
        print(one)
    
