# Required Libraries
import torch
from opacus.validators import ModuleValidator
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm.notebook import tqdm
from opacus import PrivacyEngine

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20
LR = 0.001


def train_model(epsilon, train_data_loader, batch_size=512, max_physical_batch_size=128, modelName="resnet18"):
    """
    The function train ResNet model, with epsilon-DP.
    :param epsilon:
    :param data:
    :return:
    """
    # Data handel
    # # Calculate the number of samples for 2% of the data
    # subset_length = round(0.01 * len(train_data_loader))
    # # Create a random permutation of indices and slice the first 2%
    # indices = torch.randperm(len(train_data_loader))[:subset_length]
    #
    # # Use these indices to create the subset
    # train_dataset = torch.utils.data.Subset(train_data_loader, indices)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, )

    # Model handel
    model = models.resnet18(num_classes=10)
    if modelName == "resnet18":
        model = models.resnet18(num_classes=10)

    model = ModuleValidator.fix(model)
    print(ModuleValidator.validate(model, strict=False))  # validate no errors in the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running training on device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)  # consider changing to adam

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_data_loader,
        epochs=EPOCHS,
        target_epsilon=epsilon,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        train(model, train_loader, optimizer, epoch + 1, device, privacy_engine, batch_size, max_physical_batch_size)

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
    return model


def train(model, train_loader, optimizer, epoch, device, privacy_engine, batch_size=512, max_physical_batch_size=128):
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
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
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
