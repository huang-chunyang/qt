import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from tianshou.policy import DDPGPolicy
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer
from custom_environment import FuturesTradingEnv
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

### 这个网络会产生行为输出，
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        nb_actions = 2
        init_w = 0.005
        self.hidden_rnn = 128
        self.hidden_fc1 = 256
        self.hidden_fc2 = 64
        self.hidden_fc3 = 32

        self.fc1 = nn.Linear(self.hidden_rnn, self.hidden_fc1)
        self.fc2 = nn.Linear(self.hidden_fc1, self.hidden_fc2)
        self.fc3 = nn.Linear(self.hidden_fc2, self.hidden_fc3)
        self.fc4 = nn.Linear(self.hidden_fc3, nb_actions)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        a = self.soft(x)
        return a

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.input_size = 5
        self.seq_len = 15
        self.num_layer = 2
        self.hidden_rnn = 128
        self.rnn = nn.GRU(self.input_size, self.hidden_rnn, self.num_layer, batch_first=True)
        self.cx = Variable(torch.zeros(self.num_layer, 1, self.hidden_rnn)).type(FLOAT).cuda()
        self.hx = Variable(torch.zeros(self.num_layer, 1, self.hidden_rnn)).type(FLOAT).cuda()

    def reset_hidden_state(self, done=True):
        if done == True:
            ### hx/cx：[num_layer, batch, hidden_len] ###
            self.cx = Variable(torch.zeros(self.num_layer, 1, self.hidden_rnn)).type(FLOAT).cuda()
            self.hx = Variable(torch.zeros(self.num_layer, 1, self.hidden_rnn)).type(FLOAT).cuda()
        else:
            self.cx = Variable(self.cx.data).type(FLOAT).cuda()
            self.hx = Variable(self.hx.data).type(FLOAT).cuda()
    
    def forward(self, x, hidden_states=None):
        if hidden_states == None:
            out, hx = self.rnn(x, self.hx)
            self.hx = hx
        else:
            out, hx = self.rnn(x, hidden_states)
        xh = hx[self.num_layer -1, :,:]
        return xh, hx 

## 定义了 agent 网络。该网络有两部分组成： RNN 用来分析传入的序列的特征， Actor 用来产生行为输出。
class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.rnn = RNN()
        self.actor = Actor()
        self.isTraining = True

    def select_action(self, s, noise_enable=True, decay_epslion=True):
        xh, _ = self.rnn(s)
        action = self.actor(xh)
        action = to_numpy(action.cpu()).squeeze(0)
        if noise_enable == True:
            action += self.isTraining * max(self.epsilon, 0) * np.random.randn(1)
            action = np.Softmax(action)
        return action
    
    def forward(self, s, state=None, info={}):
        action = self.select_action(s, True, False)
        return action

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