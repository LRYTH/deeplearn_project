import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 数据准备

def synthetic_data(w, b, num_examples, batch_size):
    """生成 y = Xw + b + 噪声"""
    '''w, b为真实值'''
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    y = y.reshape(-1, 1)
    dataset = TensorDataset(X, y)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_iter


