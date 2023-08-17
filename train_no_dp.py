import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np

EPOCHS = 20
LR = 0.001

def accuracy(preds, labels):
    return (preds == labels).mean()

def train_no_dp(data_loader, device):
    model = models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.to(device)
    losses = []
    top1_acc = []
    model.train()
    for epoch in range(EPOCHS):
        for i, (images, target) in enumerate(data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                )
    return model


def test_model_no_dp(model, test_loader, device):
    """
    Test model accuracy.
    :param model:
    :param test_loader:
    :param device:
    :return:
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)
