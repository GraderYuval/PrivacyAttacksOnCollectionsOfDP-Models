import random

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import copy

from training import train_model, testModel
from label_attack import MembershipInfernceAttack

class BASIC_EXPERIEMENT:

    def __init__(self, eps, ensemble_size, evaluate_score, number_of_samples=1000, batch_size=128,
                 max_physical_batch_size=128):
        self.eps = eps
        self.number_of_samples = number_of_samples
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.max_physical_batch_size = max_physical_batch_size
        self.evaluate_score = evaluate_score
        self.train_data = None
        self.test_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.ensemble = []
        self.ensemble_attacks = []
        self.evaluate = {"average": self.check_average, "max": self.check_max}

    def train_models(self):
        """
        Train n models to be epsilon-DP, using the same training set for all models.
        :param epsilon:
        :return:
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(self.ensemble_size):
            model = train_model(self.eps, self.train_data, self.batch_size, self.max_physical_batch_size)
            self.ensemble.append(model)
            acc = testModel(model)

            # new_model = torchvision.models.resnet18(pretrained=True)
            # new_model.load_state_dict(model.state_dict())
            # self.ensemble.append(new_model)
            # acc = testModel(new_model)

            print("model", i + 1, "accuracy: ", acc)

    def prepare_attack(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model in self.ensemble:
            attack = MembershipInfernceAttack(model, device)
            attack.fit(True)
            self.ensemble_attacks.append(attack)

    def check_average(self, x_sample, y_sample):
        """
        This experiment check the basic attack when take the average of the attack of the models.
        :return:
        """
        res = []
        for i in range(len(self.ensemble)):
            self.ensemble_attacks[i].attack(x_sample, y_sample)
        return np.mean(res)

    def check_max(self, x_sample, y_sample):
        res = []
        for i in range(len(self.ensemble)):
            self.ensemble_attacks[i].attack(x_sample, y_sample)
        return np.max(res)

    def get_CIFAR_ten(self):
        """
        The function download CIFAR-10 to data loader and split it to training and test data.
        :return:
        """
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV), ])

        full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Calculate the size of the dataset
        total_size = len(full_train_dataset)
        train_size = total_size // 2
        test_size = total_size - train_size

        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = \
            torch.utils.data.random_split(full_train_dataset, [train_size, test_size])

        self.train_data = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_data = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, )


    def plot_ROC_curve(self, fpr, tpr):
        # Compute the AUC (Area Under the Curve)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.semilogx()
        plt.semilogy()
        plt.xlim(1e-5, 1)
        plt.ylim(1e-5, 1)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig("fprtpr_" + self.evaluate_score + ".png")
        plt.show()


    def make_experiment(self):
        """
        The function make the experiment and produce ROC curve.
        :return:
        """
        y_true = ([1] * self.number_of_samples) + ([0] * self.number_of_samples)
        y_score = []
        for i in range(self.number_of_samples):
            x_sample, y_sample = self.train_dataset[torch.randint(len(self.train_dataset), size=(1,)).item()]
            y_score.append(self.evaluate[self.evaluate_score](x_sample, y_sample))

        for i in range(self.number_of_samples):
            x_sample, y_sample = self.test_dataset[torch.randint(len(self.test_dataset), size=(1,)).item()]
            y_score.append(self.evaluate[self.evaluate_score](x_sample, y_sample))
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        self.plot_ROC_curve(fpr, tpr)

if __name__ == '__main__':
    # first experiment. Epsilon = 2, size of the ensemble is 2 and evaluate_score is average.
    exp1 = BASIC_EXPERIEMENT(2, 2, "average")
    exp1.get_CIFAR_ten()
    print("train\n")
    exp1.train_models()
    print("prepare attack\n")
    exp1.prepare_attack()
    print("attack\n")
    exp1.make_experiment()