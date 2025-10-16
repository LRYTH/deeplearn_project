import os
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.lenet_5 import LeNet_5
from utils.accuracy import accuracy_count, evaluate_accuracy
from utils.init import init_weights


# -----------------------------
# 自定义 Dataset
# -----------------------------
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CatDogDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # 只读取 jpg 文件，并根据文件名生成标签
        self.data = []
        for fname in os.listdir(img_dir):
            if not fname.endswith('.jpg'):
                continue
            if fname.startswith('cat'):
                label = 0
            elif fname.startswith('dog'):
                label = 1
            else:
                continue
            self.data.append((fname, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert('L')  # 灰度图
        if self.transform:
            image = self.transform(image)
        else:
            # 默认 resize + tensor
            image = image.resize((28,28))
            image = transforms.ToTensor()(image)  # 输出 [1,28,28]

        return image, torch.tensor(label, dtype=torch.long)



# -----------------------------
# 数据变换（CNN用）
# -----------------------------


# -----------------------------
# 创建 Dataset 和 DataLoader
# -----------------------------
# 训练集
# CNN 变换
transform_cnn = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),  # 输出 [3,28,28]
])

train_dataset = CatDogDataset(img_dir='data/dog_and_cat/train', is_train=True, transform=None)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CatDogDataset(img_dir='data/dog_and_cat/test', csv_file='data/dog_and_cat/sampleSubmission.csv',
                             is_train=False, transform=None)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------
# 检查数据
# -----------------------------
for imgs, labels in train_loader:
    print("Batch imgs shape:", imgs.shape)  # MLP: [batch, 128*128*3]
    print("Batch labels:", labels)
    break


# 多层感知器
net = LeNet_5()

# 初始化权重
net.apply(init_weights)

# 损失函数和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# 训练
epochs = 10

train_losses, train_accs, test_accs = [], [], []
for epoch in range(1, epochs+1):
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
plt.plot(range(1, epochs + 1), train_losses, 'o-', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accs, 'o-', label='Training Accuracy')
plt.plot(range(1, epochs+1), test_accs, 'o-', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

