import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from datasets.image import get_fashion_mnist_loader
from models.mlp import MLP
from utils.accuracy import accuracy_count, evaluate_accuracy
from utils.init import init_weights
from config import Config
# 定义数据
transform = transforms.Compose([transforms.ToTensor()])
train_loader, test_loader = get_fashion_mnist_loader(batch_size=64, transforms=transform)


# 定义模型
net = MLP(784, 256, 10)

config = Config()
config.LEARNING_RATE = 0.001
# 初始化权重
net.apply(init_weights)

# 损失函数 与  优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

# 训练
train_losses, train_accs, test_accs = [], [], []
config.EPOCHS = 50
for epoch in range(1, config.EPOCHS + 1):
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

epochs = range(1, config.EPOCHS + 1)
plt.figure(figsize=[12, 5])
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'o-', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, 'o-', label='Training Accuracy')
plt.plot(epochs, test_accs, 'o-', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

