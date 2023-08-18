import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc

from label_attack import MembershipInfernceAttack
from training import train_model, testModel
import os


class BASIC_EXPERIEMENT:

    def __init__(self, eps, ensemble_size, number_of_samples=1000, batch_size=128,
                 max_physical_batch_size=128, max_grad_norm=1.2, delta=1e-5,
                 epochs=10, lr=0.001):
        self.eps = eps
        self.ensemble_size = ensemble_size
        self.number_of_samples = number_of_samples
        self.batch_size = batch_size
        self.max_physical_batch_size = max_physical_batch_size
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.epochs = epochs
        self.lr = lr
        self.train_data = None
        self.test_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.ensemble = []
        self.ensemble_attacks = []
        self.evaluate = {"average": self.check_average, "max": self.check_max}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.split_seed = np.random.randint(0, 2 ** 31 - 1)

    def train_models(self):
        """
        Train n models to be epsilon-DP, using the same training set for all models.
        :param epsilon:
        :return:
        """
        model_name = "resnet18"
        data_name = "CIFAR10"

        for i in range(self.ensemble_size):
            identifier = self.generate_model_identifier(self.eps, self.ensemble_size, self.batch_size, model_name, data_name)

            if self.model_exists(identifier, i):
                # Load model and config
                model = self.load_model_weights(train_model(self.eps, None, None, None, modelName=model_name), identifier, i)

            else:
                # Train model
                self.save_config(f"./saved_models/{identifier}_config.json")
                model = train_model(self.eps, self.train_data, self.batch_size, self.max_physical_batch_size, modelName=model_name)
                self.save_model_weights(model, identifier, i)

            self.load_config(f"./saved_models/{identifier}_config.json")
            self.ensemble.append(model)

            acc = testModel(model, self.test_data, device=self.device)
            print("model", i + 1, "accuracy: ", acc)

    def prepare_attack(self):
        for model in self.ensemble:
            attack = MembershipInfernceAttack(model, self.device)
            attack.fit(True)
            self.ensemble_attacks.append(attack)

    def check_average(self, x_sample, y_sample):
        """
        This experiment check the basic attack when take the average of the attack of the models.
        :return:
        """
        res = []
        for i in range(len(self.ensemble)):
            res.append(self.ensemble_attacks[i].attack(x_sample, y_sample))
        return np.mean(res)

    def check_max(self, x_sample, y_sample):
        res = []
        for i in range(len(self.ensemble)):
            res.append(self.ensemble_attacks[i].attack(x_sample, y_sample))
        return np.max(res)

    def get_CIFAR_ten(self):
        """
        The function download CIFAR-10 to data loader and split it to training and test data.
        :return:
        """
        if hasattr(self, 'split_seed'):
            torch.manual_seed(self.split_seed)
            np.random.seed(self.split_seed)
        else:
            raise ValueError("No split_seed available!")

        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV), ])

        full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Calculate the size of the dataset
        total_size = len(full_train_dataset)
        train_size = total_size // 2
        test_size = total_size - train_size

        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])

        self.train_data = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_data = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, )

    def plot_ROC_curve(self, fpr, tpr, evaluate_score):
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
        plt.savefig("fprtpr_" + evaluate_score + "_size_of_ensemble_" + str(self.ensemble_size) + ".png")
        plt.show()

    def seed_rng(self):
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)

    def make_experiment(self, evaluate_score):
        """
        The function make the experiment and produce ROC curve.
        :return:
        """
        self.seed_rng()
        y_true = ([1] * self.number_of_samples) + ([0] * self.number_of_samples)
        y_score = []
        for i in range(self.number_of_samples):
            x_sample, y_sample = self.train_dataset[torch.randint(len(self.train_dataset), size=(1,)).item()]
            y_score.append(self.evaluate[evaluate_score](x_sample, np.ndarray(y_sample, dtype=float)))

        for i in range(self.number_of_samples):
            x_sample, y_sample = self.test_dataset[torch.randint(len(self.test_dataset), size=(1,)).item()]
            y_score.append(self.evaluate[evaluate_score](x_sample, np.ndarray(y_sample, dtype=float)))
        print(self.ensemble_size, "\t", y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        self.plot_ROC_curve(fpr, tpr, evaluate_score)

    def save_config(self, path):
        config = {
            'eps': self.eps,
            'number_of_samples': self.number_of_samples,
            'ensemble_size': self.ensemble_size,
            'batch_size': self.batch_size,
            'max_physical_batch_size': self.max_physical_batch_size,
            'max_grad_norm': self.max_grad_norm,
            'delta': self.delta,
            'epochs': self.epochs,
            'lr': self.lr,
            'split_seed': self.split_seed
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f)

    def load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)

    def generate_model_identifier(self, eps, ensemble_size, batch_size, model_name, data_name):
        return f"model_{model_name}_data_{data_name}_eps_{eps}_ensemble_{ensemble_size}_batch_{batch_size}"

    def save_model_weights(self, model, identifier, idx):
        path = f"./saved_models/{identifier}_model_{idx}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    def load_model_weights(self, model, identifier, idx):
        path = f"./saved_models/{identifier}_model_{idx}.pth"
        model.load_state_dict(torch.load(path))
        model.eval()  # Set model to evaluation mode
        return model

    def model_exists(self, identifier, idx):
        return os.path.exists(f"./saved_models/{identifier}_model_{idx}.pth")


def run_experiment(eps, ensemble_size):
    exp = BASIC_EXPERIEMENT(eps=eps, ensemble_size=ensemble_size)

    exp.get_CIFAR_ten()
    print("train ", ensemble_size, "\n")

    exp.train_models()
    print("prepare attack ", ensemble_size, "\n")

    exp.prepare_attack()
    print("attack ", ensemble_size, "\n")
    exp.make_experiment("average")
    exp.make_experiment("max")


if __name__ == '__main__':
    epsilon = 8
    n = 2

    # Train and attack: 2*epsilon-DP model
    run_experiment(epsilon, 1)

    # Train and attack ensemble of 2: epsilon-DP model
    run_experiment(epsilon / 2, n)
