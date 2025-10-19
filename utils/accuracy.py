import torch


# 计算正确的个数
def accuracy_count(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 一个数据集在模型上的精度
def evaluate_accuracy(net, data_iter, device):
    net.eval()
    acc, total = 0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            acc += accuracy_count(net(x), y)
            total += y.numel()
    return acc / total