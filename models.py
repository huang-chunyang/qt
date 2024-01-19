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
class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.gru = nn.GRU(input_size=state_shape[1], hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, action_shape)
        self.device = device
        self.layer_norm = nn.LayerNorm(64)

    def forward(self, s, state=None, info={}):
        # print(s)
        # print(type(s))
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32).to(self.device)
        if state is not None:
            state = state.to(self.device)
        s, h = self.gru(s, state or None)
        s = self.layer_norm(s[:, -1])
        return self.fc(s), None

from tianshou.trainer import BaseTrainer

def my_callback(trainer: BaseTrainer, epoch: int, env_step: int, gradient_step: int, **kwargs):
    # 这里我们假设您想要输出的信息保存在Collector的统计信息中
    train_stats = trainer.train_collector.collect_info
    test_stats = trainer.test_collector.collect_info

    print(f"Epoch {epoch}:")
    print("Train Collector Info:")
    for key, value in train_stats.items():
        print(f"  {key}: {value}")

    print("Test Collector Info:")
    for key, value in test_stats.items():
        print(f"  {key}: {value}")

# 定义环境和相关参数
env = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC", data_mode="train") for _ in range(2)])


train_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC", data_mode="train") for _ in range(10)])
test_envs = DummyVectorEnv([lambda: FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC", data_mode="test") for _ in range(10)])

# 创建网络实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tianshou.policy import DQNPolicy
from torch.optim import Adam

action_space = env.action_space[0]
action_shape = action_space.shape or action_space.n
state_shape = env.observation_space[0]
print(action_shape, state_shape)
net = Net(state_shape, action_shape, device).to(device)
optim = Adam(net.parameters(), lr=1e-3)
policy = DQNPolicy(net, optim, 
                   # gamma=0.99, n_step=3, target_update_freq=320
                  )
policy.set_eps(0.5)
# 数据收集器
train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=50000, buffer_num=10))
test_collector = Collector(policy, test_envs)

# 训练
result = offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=2400, step_per_collect=10,
    update_per_step=0.1, episode_per_test=10, batch_size=64,
    # callback=my_callback
)
for one in result:
    print(one)
    # print(epoch_stat)
    # print(info)