from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_fashion_mnist_loader(batch_size, transforms):
    train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms)
    test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_mnist_loader(batch_size, transforms):
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = get_mnist_loader(batch_size=64, transforms=transforms.ToTensor())
for X, y in train_loader:
    print(X.shape)
    print(y.shape)
    break