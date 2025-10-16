import os

import torch
from torch import nn
from datasets.linear_datasets import synthetic_data
from models.linear import Linear
from utils.init import init_weights
from torch.utils.tensorboard import SummaryWriter


batch_size = 100
num_examples = 1000

true_w = torch.tensor([2, -3.4])
true_b = 4.2

data_iter = synthetic_data(true_w, true_b, num_examples, batch_size)

writer = SummaryWriter('logs/linear')


net = Linear(2, 1)
# 初始化模型参数
net.apply(init_weights)

# 损失函数 优化算法
loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# 训练
train_losses = []
epochs = 1000
for epoch in range(1, epochs + 1):
    total_loss = 0
    count = 0
    for X, y in data_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += l.item()
        count += 1
    avg_loss = total_loss / count
    writer.add_scalar('loss', avg_loss, epoch)
    print(f'epoch {epoch}, loss {avg_loss:.3f}')

writer.close()
torch.save(net.state_dict(), f'checkpoints/linear/linear_model.pth')