import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from tianshou.policy import DDPGPolicy
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer

# 确保您的自定义环境已经导入
from custom_environment import FuturesTradingEnv

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.gru = nn.GRU(input_size=state_shape[1], hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)  # 输出大小为2，对应两个动作的概率

    def forward(self, s, state=None, info={}):
        s, h = self.gru(s, state)
        return nn.functional.softmax(self.fc(s[:, -1]), dim=-1), h

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.gru = nn.GRU(input_size=state_shape[1] + 2, hidden_size=64, batch_first=True)  # 加2是因为动作是one-hot编码
        self.fc = nn.Linear(64, 1)

    def forward(self, s, a, state=None, info={}):
        # 将动作转换为one-hot编码
        a_one_hot = torch.nn.functional.one_hot(a.long(), num_classes=2)
        x = torch.cat([s, a_one_hot], dim=-1)
        x, h = self.gru(x, state)
        return self.fc(x[:, -1]), h
class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.gru = nn.GRU(input_size=state_shape[1], hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, action_shape)
        self.device = device

    def forward(self, s, state=None, info={}):
        # print(s)
        # print(type(s))
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32).to(self.device)
        if state is not None:
            state = state.to(self.device)
        s, h = self.gru(s, state or None)
        return self.fc(s[:, -1]), None

# 定义环境和相关参数
env = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC") for _ in range(2)])
state_shape = env.observation_space[0]
action_shape = env.action_space[0]
print(state_shape)
print(action_shape)
train_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC") for _ in range(10)])
test_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC") for _ in range(10)])

# 创建网络实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tianshou.policy import DQNPolicy
from torch.optim import Adam

net = Net(state_shape, 2, device).to(device)
optim = Adam(net.parameters(), lr=1e-3)
policy = DQNPolicy(net, optim, 
                   # gamma=0.99, n_step=3, target_update_freq=320
                  )

# 数据收集器
train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=50000, buffer_num=10))
test_collector = Collector(policy, test_envs)

# 训练
result = offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=1000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=10, batch_size=64,
)