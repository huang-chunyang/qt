import numpy as np
import torch

from tianshou.trainer import BaseTrainer

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

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