import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from training import train_model, testModel
from LiRA import LiRA


class BASIC_EXPERIEMENT:

    def __init__(self, eps, ensemble_size, number_of_samples=25, batch_size=64,
                 max_physical_batch_size=128):
        self.eps = eps
        self.number_of_samples = number_of_samples
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.max_physical_batch_size = max_physical_batch_size
        self.full_train_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.ensemble = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_models(self):
        """
        Train n models to be epsilon-DP, using the same training set for all models.
        :param epsilon:
        :return:
        """
        train_data = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
        test_data = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, )
        for i in range(self.ensemble_size):
            # train with DP
            model = train_model(self.eps, train_data, self.batch_size, self.max_physical_batch_size)
            acc = testModel(model, test_data, device=self.device)
            print("model", i + 1, "accuracy: ", acc)
            model = model.to('cpu')
            self.ensemble.append(model)
            torch.cuda.empty_cache()

    def check_attack(self, index):
        """
        This experiment check LiRA attack when take the average of the attack of the models.
        :return:
        """
        res = []
        lira = LiRA(self.eps, self.full_train_dataset, index, self.device, N=16, batch_size=self.batch_size)
        lira.prepare_attack()
        for i in range(len(self.ensemble)):
            ret = lira.attack(self.ensemble[i])
            res.append(ret)

        return np.mean(res), np.max(res)

    def get_CIFAR_ten(self):
        """
        The function download CIFAR-10 to data loader and split it to training and test data.
        :return:
        """
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV), ])

        full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.full_train_dataset = full_train_dataset

        # Calculate the size of the dataset
        total_size = len(full_train_dataset)
        train_size = total_size // 2
        test_size = total_size - train_size

        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = \
            torch.utils.data.random_split(full_train_dataset, [train_size, test_size])

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
        plt.savefig("changed_" + evaluate_score + "_size_of_ensemble_" + str(self.ensemble_size) + "_eps_" +
                    str(self.eps) + ".png")
        plt.show()

    def make_experiment(self):
        """
        The function make the experiment and produce ROC curve.
        :return:
        """
        y_true = ([1] * self.number_of_samples) + ([0] * self.number_of_samples)
        y_average = []
        y_max = []
        for i in range(self.number_of_samples):
            index = torch.randint(len(self.train_dataset), size=(1,))
            index = self.train_dataset.indices[index]
            average, m = self.check_attack(index)
            y_average.append(average)
            y_max.append(m)

        for i in range(self.number_of_samples):
            index = torch.randint(len(self.test_dataset), size=(1,))
            index = self.test_dataset.indices[index]
            average, m = self.check_attack(index)
            y_average.append(average)
            y_max.append(m)

        fpr, tpr, thresholds = roc_curve(y_true, y_average, pos_label=1)
        print("param: ", self.ensemble_size, "\t", fpr, tpr, thresholds, "\n", y_average, "\n", y_max, "\n")
        self.plot_ROC_curve(fpr, tpr, "average")
        self.plot_ROC_curve(fpr, tpr, "max")


if __name__ == '__main__':
    for i in range(2, 11):
        torch.cuda.empty_cache()
        exp = BASIC_EXPERIEMENT(8, i, number_of_samples=5)
        exp.get_CIFAR_ten()
        print("train ", i, "\n")
        exp.train_models()
        print("attack ", i, "\n")
        exp.make_experiment()