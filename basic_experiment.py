import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

from training import train_model, testModel
# from label_attack import attack

class BASIC_EXPERIEMENT:

    def __init__(self, eps, ensemble_size, evaluate_score, number_of_samples=1000, batch_size=512, max_physical_batch_size=128):
        self.eps = eps
        self.number_of_samples = number_of_samples
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.max_physical_batch_size = max_physical_batch_size
        self.evaluate_score = evaluate_score
        self.train_data = None
        self.test_data = None
        self.ensemble = []
        self.evaluate = {"average": self.check_average, "max": self.check_max}

    def train_models(self):
        """
        Train n models to be epsilon-DP, using the same training set for all models.
        :param epsilon:
        :return:
        """
        for i in range(self.ensemble_size):
            model = train_model(self.eps, self.train_data, self.batch_size, self.max_physical_batch_size)
            acc = testModel(model, self.test_data)
            self.ensemble.append(model)
            print("model", i + 1, "accuracy: ", acc)

    def check_average(self, sample):
        """
        This experiment check the basic attack when take the average of the attack of the models.
        :return:
        """
        res = [attack(model, sample, self.train_data) for model in self.ensemble]
        return np.mean(res)

    def check_max(self, sample):
        res = [attack(model, sample, self.train_data) for model in self.ensemble]
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
        train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])

        self.train_data = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, )
        self.test_data = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, )

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
        plt.savefig("/tmp/fprtpr " + self.evaluate_score + ".png")
        plt.show()


    def make_experiment(self):
        """
        The function make the experiment and produce ROC curve.
        :return:
        """
        sampled_indices = np.random.choice(self.train_data.index, size=self.number_of_samples, replace=False)
        true_sample = self.train_data.loc[sampled_indices]
        sampled_indices = np.random.choice(self.test_data.index, size=self.number_of_samples, replace=False)
        false_sample = self.test_data.loc[sampled_indices]
        y_true = ([1] * self.number_of_samples) + ([0] * self.number_of_samples)
        y_score = []
        for i in range(self.number_of_samples):
            y_score.append(self.evaluate[self.evaluate_score](self.ensemble, true_sample[i], self.train_data))

        for i in range(self.number_of_samples):
            y_score.append(self.evaluate[self.evaluate_score](self.ensemble, false_sample[i], self.train_data))

        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        self.plot_ROC_curve(fpr, tpr)

if __name__ == '__main__':
    # first experiment. Epsilon = 2, size of the ensemble is 2 and evaluate_score is average.
    exp1 = BASIC_EXPERIEMENT(2, 2, "average")
    exp1.get_CIFAR_ten()
    exp1.train_models()
    exp1.make_experiment()
