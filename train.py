import torch.cuda
from torch.utils.data import DataLoader
from torchvision import datasets
from torchsummary import summary
from torchvision.transforms import ToTensor

import utils
from MyNetUtils import MyNet
from torch import nn
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--learning_rate', default=1e-3, type=int, help='world size')

    args = parser.parse_args()

    return args

def train_loop(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(dataloader, model, loss_fn,device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():

    args = vars(parse_args())
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    """Datasets"""

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    val_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    """构建DataLoader"""
    train_dataloader = DataLoader(training_data, batch_size = args["batch_size"])
    test_dataloader = DataLoader(val_data, batch_size = args["batch_size"])

    """实例化网络"""
    model = MyNet()

    """定义损失函数"""
    loss_fn = nn.CrossEntropyLoss()

    """定义优化器"""
    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])

    """开始训练"""
    for t in range(args["epoch"]):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer,device)
        val_loop(test_dataloader, model, loss_fn,device)
        utils.save_checkpoint(model,optimizer,args["learning_rate"],t,"logs")

    print("Done!")


if __name__ == '__main__':
    main()
