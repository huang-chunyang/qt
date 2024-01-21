import torch
import torch.nn as nn
import numpy as np
<<<<<<< HEAD
from torch.distributions import Normal
from tianshou.policy import DDPGPolicy
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer

# 确保您的自定义环境已经导入
from custom_environment import FuturesTradingEnv

# 定义Actor网络
class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.gru = nn.GRU(input_size=state_shape[1], hidden_size=640, batch_first=True)
        self.fc = nn.Linear(640, action_shape)
        self.device = device
        self.layer_norm = nn.LayerNorm(640)

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
=======
from torch.autograd import Variable
from utils import fanin_init, to_numpy, FLOAT

>>>>>>> origin/bak
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### 收集 RNN 网络产生的状态信息，输出为一个行为信号。 ###
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        nb_actions = 2
        init_w = args.init_w
        self.hidden_rnn = args.hidden_rnn
        self.hidden_fc1 = args.hidden_fc1
        self.hidden_fc2 = args.hidden_fc2
        self.hidden_fc3 = args.hidden_fc3

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
    def __init__(self, args):
        super(RNN, self).__init__()
        self.input_size = args.input_size
        self.seq_len = args.seq_len
        self.num_layer = args.num_rnn_layer
        self.hidden_rnn = args.hidden_rnn
        self.bsize = args.bsize
        self.rnn = nn.GRU(self.input_size, self.hidden_rnn, self.num_layer, batch_first=True).to(device)
        self.cx = Variable(torch.zeros(self.num_layer, self.bsize, self.hidden_rnn)).type(FLOAT).cuda()
        self.hx = Variable(torch.zeros(self.num_layer, self.bsize, self.hidden_rnn)).type(FLOAT).cuda()

<<<<<<< HEAD
# 训练
result = offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=240, step_per_collect=10,
    update_per_step=0.1, episode_per_test=10, batch_size=64,
    # callback=my_callback
)
# for one in result:
#     print(one)
    # print(epoch_stat)
    # print(info)
=======
    def reset_hidden_state(self, done=True):
        if done == True:
            ### hx/cx：[num_layer, batch, hidden_len] ###
            self.cx = Variable(torch.zeros(self.num_layer, self.bsize, self.hidden_rnn)).type(FLOAT).cuda()
            self.hx = Variable(torch.zeros(self.num_layer, self.bsize, self.hidden_rnn)).type(FLOAT).cuda()
        else:
            self.cx = Variable(self.cx.data).type(FLOAT).cuda()
            self.hx = Variable(self.hx.data).type(FLOAT).cuda()
    
    def forward(self, x, hidden_states=None):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(device)
        if hidden_states == None:
            out, hx = self.rnn(x, self.hx)
            self.hx = hx
        else:
            out, hx = self.rnn(x, hidden_states)
        ## 使用倒数第二层的输出作为最终的输出
        xh = hx[self.num_layer -1, :,:]
        return xh, hx 

## 定义了 agent 网络。该网络有两部分组成： RNN 用来分析传入的序列的特征， Actor 用来产生行为输出。
class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rnn = RNN(args).to(device)
        self.actor = Actor(args).to(device)
        self.isTraining = True

    def select_action(self, s, noise_enable=True, decay_epslion=True):
        xh, _ = self.rnn(s)
        action = self.actor(xh)
        # action = to_numpy(action.cpu()).squeeze(0)
        # if noise_enable == True:
        #     action += self.isTraining * max(self.epsilon, 0) * np.random.randn(1)
        #     action = np.Softmax(action)
        return action
    
    def forward(self, s, state=None, info={}):
        action = self.select_action(s, True, False)
        return action
>>>>>>> origin/bak
