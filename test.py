#%%
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch
class AnimalDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        # 遍历文件夹下的所有图片
        for fname in os.listdir(img_dir):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                # 根据文件名前缀提取标签
                if fname.startswith('cat'):
                    label = 0
                elif fname.startswith('dog'):
                    label = 1
                else:
                    continue
                self.data.append((fname, label))

    def __getitem__(self, idx):
        fname, label = self.data[idx] # 获取图片名和标签
        img_path = os.path.join(self.img_dir, fname) # 构建图片路径
        image = Image.open(img_path).convert('RGB')  # 转为RGB # 读取图片
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(image)
        return image, label

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        df = df.sort_values(by="id")  #  按 id 排序，防止顺序乱
        # 保存每张图片的id和label
        self.data = [(str(row['id']) + '.jpg', int(row['label'])) for _, row in df.iterrows()]

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert('RGB')  # RGB彩色图像
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(image)
        return image, label

    def __len__(self):
        return len(self.data)

#%%
# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载数据集
train_dataset = AnimalDataset('data/dog_and_cat/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TestDataset('data/dog_and_cat/test', csv_path='data/dog_and_cat/sampleSubmission.csv', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# # 取一批数据看看
# images, labels = next(iter(train_loader))
# print(images)   # torch.Size([32, 3, 28, 28])
# print(labels[:10])

# 取一批数据看看
images, labels = next(iter(test_loader))
print(images)
print(labels[:10])
#%%
from matplotlib import pyplot as plt
from utils.accuracy import evaluate_accuracy
import torch
from torch import nn
from utils.init import init_weights
from models.alexnet import AlexNet

# 定义网络
net = AlexNet()
net.apply(init_weights)
net.to(device)
# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# 训练
epochs = 10
train_losses, train_accs, test_accs = [], [], []
for epoch in range(1, epochs+1):
    net.train()
    total_loss, total_acc, total_count = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
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
    test_acc = evaluate_accuracy(net, test_loader, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f'Epoch {epoch}, Loss {train_loss:.4f}, Acc {train_acc:.4f}, Test Acc {test_acc:.4f}')

plt.figure(figsize=[12, 5])
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'o-', label='Training Loss')
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


#%%
