import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import fanin_init, to_numpy, FLOAT

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