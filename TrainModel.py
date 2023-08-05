# Required Libraries
#TODO DELETE THIS CLASS
import torch
import torchvision.transforms as transforms
from opacus.validators import ModuleValidator
from torchvision import models
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.optim as optim
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm.notebook import tqdm

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128
DATA_ROOT = '../data'


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
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

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )


def testModel(model, test_loader, device):
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


def accuracy(preds, labels):
    return (preds == labels).mean()


if __name__ == '__main__':
    # Data handel
    # TODO why are we using this?
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV), ])

    full_train_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    # Calculate the number of samples for 2% of the data
    subset_length = round(0.01 * len(full_train_dataset))
    # Create a random permutation of indices and slice the first 2%
    indices = torch.randperm(len(full_train_dataset))[:subset_length]
    # Use these indices to create the subset
    train_dataset = torch.utils.data.Subset(full_train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, )


    test_dataset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,)
    print("finished preparing data")

    # Model handel
    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model)
    print(ModuleValidator.validate(model, strict=False))  # validate no errors in the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running training on device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LR)  # consider changing to adam

    from opacus import PrivacyEngine

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        train(model, train_loader, optimizer, epoch + 1, device)

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

    top1_acc = test(model, test_loader, device)
