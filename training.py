# Required Libraries
import torch
from opacus.validators import ModuleValidator
from torchvision import models

# import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm
from opacus import PrivacyEngine


def create_model(modelName="resnet18"):
    model = models.resnet18(num_classes=10)
    if modelName == "resnet18":
        model = models.resnet18(num_classes=10)

    model = ModuleValidator.fix(model)
    print(ModuleValidator.validate(model, strict=False))  # validate no errors in the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running training on device: {device}")
    model = model.to(device)
    return model


def train_model(epsilon, train_data_loader, batch_size=512, max_physical_batch_size=128,
                modelName="resnet18", max_grad_norm=1.2, epochs=10, lr=0.001, delta=1e-5):
    """
    The function train ResNet model, with epsilon-DP.
    :param epsilon:
    :param data:
    :return:
    """
    # Model handel

    model = create_model(modelName=modelName)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=lr)  # consider changing to adam

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_data_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    for epoch in tqdm(range(epochs)):
        train(model, train_loader, optimizer, epoch + 1, device, privacy_engine, batch_size, max_physical_batch_size, delta)

    del privacy_engine
    print(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")
    return model


def train(model, train_loader, optimizer, epoch, device, privacy_engine, batch_size, max_physical_batch_size, delta):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []
    torch.cuda.empty_cache()

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
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
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(delta=delta)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {delta})"
                )


def testModel(model, test_loader, device="cpu"):
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


def accuracy(preds, labels):
    return (preds == labels).mean()
