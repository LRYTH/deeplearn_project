import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms

from config import Config
from datasets.image import get_fashion_mnist_loader
from models.mlp import MLP_dropout
from utils.accuracy import accuracy_count, evaluate_accuracy
from utils.init import init_weights

config = Config()
config.LEARNING_RATE = 0.0001
config.DROPOUT_RATE = 0.5
config.BATCH_SIZE = 64

# 数据
trans = transforms.Compose([transforms.ToTensor()])
train_loader, test_loader = get_fashion_mnist_loader(batch_size=config.BATCH_SIZE, transforms=trans)



# 网络
net = MLP_dropout(784, 256, 10, config)

# 初始化权重
net.apply(init_weights)

# 损失函数与优化器
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)


# 训练
config.EPOCHS = 50
train_losses, train_accs, test_accs = [], [], []
for epoch in range(1, config.EPOCHS+1):
    net.train()
    total_loss, total_acc, total_count = 0, 0, 0
    for X, y in train_loader:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += l.item() * y.numel()
        total_acc += accuracy_count(y_hat, y)
        total_count += y.numel()

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    test_acc = evaluate_accuracy(net, test_loader)

    train_losses.append(avg_loss)
    train_accs.append(avg_acc)
    test_accs.append(test_acc)
    print(f'Epoch {epoch}, Loss {avg_loss:.4f}, Acc {avg_acc:.4f}, Test Acc {test_acc:.4f}')

# 可视化
plt.figure(figsize=[12, 5])
plt.subplot(1, 2, 1)
plt.plot(range(1, config.EPOCHS+1), train_losses, 'o-', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(1, config.EPOCHS+1), train_accs, 'o-', label='Training Accuracy')
plt.plot(range(1, config.EPOCHS+1), test_accs, 'o-', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()