import torch
from torch import nn
from matplotlib import pyplot as plt
from datasets.image import get_fashion_mnist_loader
from torchvision import transforms
from models.softmax import Softmax
from utils.init import init_weights
from utils.accuracy import evaluate_accuracy, accuracy_count

# 数据准备
batch_size = 256
trans = transforms.Compose([transforms.ToTensor()])
train_loader, test_loader = get_fashion_mnist_loader(batch_size, trans)

num_inputs = 784
num_outputs = 10


net = Softmax(num_inputs, num_outputs)

# 这是初始化权重函数  可以直接全部设置神经网络中的线性回归的权重 偏置默认是 0 不写

net.apply(init_weights)

# 损失函数里带有softmax 所以不需要自己手动指定
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)




train_losses, train_accs, test_accs = [], [], []

num_epochs = 50
for epoch in range(num_epochs):
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

    print(f'Epoch {epoch + 1}, Loss {avg_loss:.4f}, Acc {avg_acc:.4f}, Test Acc {test_acc:.4f}')

epochs = range(1, num_epochs + 1)
plt.figure(figsize=[12, 5])

# 损失函曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'o-', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, 'o-', label='Training Accuracy')
plt.plot(epochs, test_accs, 'o-', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()