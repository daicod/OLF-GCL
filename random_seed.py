import numpy as np
import torch
import random
import os


def setup_seed(seed):
    np.random.seed(seed) # numpy 的设置
    random.seed(seed)  # python random module
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了使得hash随机化，使得实验可以复现
    torch.manual_seed(seed) # 为cpu设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU为，所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False # 设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。
        torch.backends.cudnn.deterministic = True # 每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，
                                                    # 应该可以保证每次运行网络的时候相同输入的输出是固定的