import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

from config import Config
from datasets.image import get_mnist_loader
from models.googlenet import GoogLeNet
from utils.accuracy import evaluate_accuracy
from utils.init import init_weights

# 定义数据
config = Config()
config.LEARNING_RATE = 0.001
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)),
                                 transforms.Resize((224, 224))])
train_loader, test_loader = get_mnist_loader(batch_size=config.BATCH_SIZE, transforms=transforms)

# 网络
net = GoogLeNet()


# # 输出调度方式
# X = torch.rand(size=(1, 1, 28, 28))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)

# 初始化权重
net.apply(init_weights)

# 损失函数优化器
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

train_losses, train_accs, test_accs = [], [], []
config.EPOCHS = 30
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
        total_acc += (y_hat.argmax(dim=1) == y).sum().item()
        total_count += y.numel()

    train_acc = total_acc / total_count
    train_loss = total_loss / total_count
    test_acc = evaluate_accuracy(net, test_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f'Epoch {epoch}, Loss {train_loss:.4f}, Acc {train_acc:.4f}, Test Acc {test_acc:.4f}')

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


