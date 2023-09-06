import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from art.estimators.classification.pytorch import PyTorchClassifier
import torch.optim as optim
from torch import nn
from Attack import Attack

from training import train_model, testModel, create_model
import os

THRESHOLD_PATH = "threshold/"
RESULTS_PATH = "results/"
INDIVIDUAL_RESULTS_PATH = "models_results/"

class BASIC_EXPERIEMENT:

    def __init__(self, eps, ensemble_size, attack_threshold, number_of_samples=1000, batch_size=64,
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
        self.attack_train_size = 1500
        self.attack_test_size = 1500
        self.attack_threshold = attack_threshold
        self.ensemble = []
        self.ensemble_attacks = []
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
            identifier = self.generate_model_identifier(self.eps, self.ensemble_size, self.batch_size, model_name,
                                                        data_name)

            if self.model_exists(identifier, i):
                # Load model and config
                model = self.load_model_weights(create_model(), identifier, i)

            else:
                # Train model
                self.save_config(f"./saved_models/{identifier}_config.json")
                model = train_model(self.eps, self.train_data, self.batch_size, self.max_physical_batch_size,
                                    modelName=model_name)
                self.save_model_weights(model, identifier, i)

            self.load_config(f"./saved_models/{identifier}_config.json")
            self.ensemble.append(model)

            acc = testModel(model, self.test_data, device=self.device)
            print("model", i + 1, "accuracy: ", acc)

    def convert_x_sample(self, x):
        x = torch.from_numpy(x)
        x = torch.permute(x, (0, 3, 1, 2))
        return torch.Tensor.numpy(x)

    def prepare_attack(self):
        x_train = self.train_dataset.dataset.data[1:self.attack_train_size]
        x_train = self.convert_x_sample(x_train)
        x_test = self.test_dataset.dataset.data[1:self.attack_test_size]
        x_test = self.convert_x_sample(x_test)
        y_train = np.ndarray(shape=(self.attack_train_size - 1,), offset=np.float_().itemsize, dtype=float,
                             buffer=np.array(self.train_dataset.dataset.targets[:self.attack_train_size]))
        y_test = np.ndarray(shape=(self.attack_train_size - 1,), offset=np.float_().itemsize, dtype=float,
                            buffer=np.array(self.test_dataset.dataset.targets[:self.attack_test_size]))
        file_name = f"{self.eps}_{self.ensemble_size}"
        file_path = os.path.join(THRESHOLD_PATH, file_name)

        for model in self.ensemble:
            art_model = PyTorchClassifier(model, loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters()),
                                          channels_first=True, input_shape=(3, 32, 32,), nb_classes=10)
            self.ensemble_attacks.append(LabelOnlyDecisionBoundary(art_model))
            if len(self.attack_threshold) < len(self.ensemble_attacks):
                self.ensemble_attacks[-1].calibrate_distance_threshold\
                    (x_train, y_train, x_test, y_test)
                with open(file_path, 'a') as file:
                    file.write(str(self.ensemble_attacks[-1].distance_threshold_tau) + "\n")

            else:
                self.ensemble_attacks[-1].distance_threshold_tau = self.attack_threshold[len(self.ensemble_attacks) - 1]

    def attack(self, x_sample, y_sample):
        """
        This experiment check the basic attack when take the average of the attack of the models.
        :return:
        """
        res = []
        for i in range(len(self.ensemble)):
            prob = self.ensemble_attacks[i].infer(x_sample, y_sample, probabilities=True)
            res.append(prob[0][1])
        return np.max(res), np.mean(res)

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

        '''
        NADAV
        full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Calculate the size of the dataset
        total_size = len(full_train_dataset)
        train_size = total_size // 2
        test_size = total_size - train_size

        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_train_dataset,
                                                                              [train_size, test_size])

        self.train_data = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_data = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, )
        '''
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)


        self.train_data = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
        )


        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

        self.test_data = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


    def seed_rng(self):
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)

    def read_results_from_file(self):
        y_average = []
        y_max = []

        file_name_average = f"results_eps_{self.eps}_ensemble_{self.ensemble_size}_average"
        file_path_average = os.path.join(RESULTS_PATH, file_name_average)
        if os.path.exists(file_path_average):
            with open(file_path_average, 'r') as file:
                for line in file:
                    if line != "\n":
                        y_average.append(float(line))

        file_name_max = f"results_eps_{self.eps}_ensemble_{self.ensemble_size}_max"
        file_path_max = os.path.join(RESULTS_PATH, file_name_max)
        if os.path.exists(file_path_max):
            with open(file_path_max, 'r') as file:
                for line in file:
                    if line != "\n":
                        y_max.append(float(line))
        return y_average, y_max
    def make_experiment(self):
        """
        The function make the experiment and produce ROC curve.
        :return:
        """
        file_name_average = f"results_eps_{self.eps}_ensemble_{self.ensemble_size}_average"
        file_path_average = os.path.join(RESULTS_PATH, file_name_average)
        file_name_max = f"results_eps_{self.eps}_ensemble_{self.ensemble_size}_max"
        file_path_max = os.path.join(RESULTS_PATH, file_name_max)
        self.seed_rng()
        y_average, y_max = self.read_results_from_file()
        for i in range(self.number_of_samples):
            x_sample, y_sample = self.train_dataset[torch.randint(len(self.train_dataset), size=(1,)).item()]
            if i < len(y_average):
                continue
            y_sample = np.ndarray(shape=(1,), buffer=np.array([0., y_sample]), offset=np.float_().itemsize,
                                  dtype=float)
            m, a = self.attack(torch.Tensor.numpy(torch.unsqueeze(x_sample, dim=0)), y_sample)
            y_average.append(a)
            y_max.append(m)
            with open(file_path_average, 'a') as file_average, open(file_path_max, 'a') as file_max:
                file_average.write(str(a) + "\n")
                file_max.write(str(m) + "\n")

        for i in range(self.number_of_samples):
            x_sample, y_sample = self.test_dataset[torch.randint(len(self.test_dataset), size=(1,)).item()]
            if (i + self.number_of_samples) < len(y_average):
                continue
            y_sample = np.ndarray(shape=(1,), buffer=np.array([0., y_sample]), offset=np.float_().itemsize,
                                  dtype=float)
            m, a = self.attack(torch.Tensor.numpy(torch.unsqueeze(x_sample, dim=0)), y_sample)
            y_average.append(a)
            y_max.append(m)
            with open(file_path_average, 'a') as file_average, open(file_path_max, 'a') as file_max:
                file_average.write(str(a) + "\n")
                file_max.write(str(m) + "\n")

        if self.ensemble_size > 1:
            return y_average, y_max
        else:
            return y_max

    def attack_individual_model(self):
        d = {1: 0, 2: 0, 3: 2, 4: 5}
        self.seed_rng()
        for j in range(self.ensemble_size):
            file_name_inidic = f"results_eps_{self.eps}_number_{j + d[self.ensemble_size]}_indicator_true"
            file_name_prob = f"results_eps_{self.eps}_number_{j + d[self.ensemble_size]}_prob_true"
            file_path_inidic = os.path.join(INDIVIDUAL_RESULTS_PATH, file_name_inidic)
            file_path_prob = os.path.join(INDIVIDUAL_RESULTS_PATH, file_name_prob)
            num_lines = 0
            if os.path.exists(file_path_inidic):
                with open(file_path_inidic, "r") as f:
                    num_lines = sum(1 for _ in f)

            for i, (x_sample, y_sample) in enumerate(self.train_data):
                if i * self.batch_size > 1000:
                    break
                if i * self.batch_size < num_lines:
                    continue
                y_sample = torch.Tensor.numpy(y_sample)
                x_sample = torch.Tensor.numpy(x_sample)
                indic, prob = self.ensemble_attacks[j].infer(x_sample, y_sample)
                self.write_to_files(indic, prob, file_path_inidic, file_path_prob)

            file_name_inidic = f"results_eps_{self.eps}_number_{j + d[self.ensemble_size]}_indicator_false"
            file_name_prob = f"results_eps_{self.eps}_number_{j + d[self.ensemble_size]}_prob_false"
            file_path_inidic = os.path.join(INDIVIDUAL_RESULTS_PATH, file_name_inidic)
            file_path_prob = os.path.join(INDIVIDUAL_RESULTS_PATH, file_name_prob)
            num_lines = 0
            if os.path.exists(file_path_inidic):
                with open(file_path_inidic, "r") as f:
                    num_lines = sum(1 for _ in f)

            for i, (x_sample, y_sample) in enumerate(self.test_data):
                if i * self.batch_size > 1000:
                    break
                if i * self.batch_size < num_lines:
                    continue
                y_sample = torch.Tensor.numpy(y_sample)
                x_sample = torch.Tensor.numpy(x_sample)
                indic, prob = self.ensemble_attacks[j].infer(x_sample, y_sample)
                self.write_to_files(indic, prob, file_path_inidic, file_path_prob)

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
        torch.save(model._module.state_dict(), path)

    def load_model_weights(self, model, identifier, idx):
        path = f"./saved_models/{identifier}_model_{idx}.pth"
        d = OrderedDict()
        for k, v in torch.load(path, map_location=self.device).items():
            if k[:8] == "_module.":
                d[k[8:]] = v
            else:
                d[k] = v
        model.load_state_dict(d)
        model.eval()  # Set model to evaluation mode
        return model

    def model_exists(self, identifier, idx):
        return os.path.exists(f"./saved_models/{identifier}_model_{idx}.pth")

    def prepare_attack_individual_model(self):
        x_train = self.train_dataset.dataset.data[1:self.attack_train_size]
        x_train = self.convert_x_sample(x_train)
        x_test = self.test_dataset.dataset.data[1:self.attack_test_size]
        x_test = self.convert_x_sample(x_test)
        y_train = np.ndarray(shape=(self.attack_train_size - 1,), offset=np.float_().itemsize, dtype=float,
                             buffer=np.array(self.train_dataset.dataset.targets[:self.attack_train_size]))
        y_test = np.ndarray(shape=(self.attack_train_size - 1,), offset=np.float_().itemsize, dtype=float,
                            buffer=np.array(self.test_dataset.dataset.targets[:self.attack_test_size]))
        file_name = f"{self.eps}_{self.ensemble_size}"
        file_path = os.path.join(THRESHOLD_PATH, file_name)

        for i in range(self.ensemble_size):
            model = self.ensemble[i]
            art_model = PyTorchClassifier(model, loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters()),
                                          channels_first=True, input_shape=(3, 32, 32,), nb_classes=10)
            self.ensemble_attacks.append(Attack(art_model))
            if len(self.attack_threshold) < len(self.ensemble_attacks):
                self.ensemble_attacks[-1].calibrate_distance_threshold \
                    (x_train, y_train, x_test, y_test)
                with open(file_path, 'a') as file:
                    file.write(str(self.ensemble_attacks[-1].distance_threshold_tau) + "\n")

            else:
                self.ensemble_attacks[-1].distance_threshold_tau = self.attack_threshold[len(self.ensemble_attacks) - 1]

    def write_to_files(self, indic, prob, file_path_inidic, file_path_prob):
        with open(file_path_inidic, 'a') as file_indic, open(file_path_prob, 'a') as file_prob:
            for i in range(len(indic)):
                file_indic.write(str(indic[i]) + "\n")
                file_prob.write(str(prob[i]) + "\n")

def plot_ROC_curve(y_average, y_max, y_single, eps, ensemble_size, number_of_samples):
    plt.figure(figsize=(8, 6))
    y_true = ([1] * number_of_samples) + ([0] * number_of_samples)
    # plot average roc curve
    fpr_a, tpr_a, thresholds_a = roc_curve(y_true, y_average, pos_label=1)
    roc_auc_a = auc(fpr_a, tpr_a)
    plt.plot(fpr_a, tpr_a, color='b', lw=2, label=f'average (AUC = {roc_auc_a:.2f})')
    # plot max roc curve
    fpr_m, tpr_m, thresholds_m = roc_curve(y_true, y_max, pos_label=1)
    roc_auc_m = auc(fpr_m, tpr_m)
    plt.plot(fpr_m, tpr_m, color='g', lw=2, label=f'max (AUC = {roc_auc_m:.2f})')
    # plot single model roc curve
    fpr_s, tpr_s, thresholds_s = roc_curve(y_true, y_single, pos_label=1)
    roc_auc_s = auc(fpr_s, tpr_s)
    plt.plot(fpr_s, tpr_s, color='r', lw=2, label=f'single model (AUC = {roc_auc_s:.2f})')

    # Plot the ROC curve
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f"ROC Curve epsilon - {eps} ensemble of - {ensemble_size}")
    plt.legend(loc='lower right')
    # plt.savefig(f"epsilon_{eps}_size_of_ensemble_{ensemble_size}.png")
    plt.show()

def manage_threshold(eps, ensemble_size):
    file_name = f"{eps}_{ensemble_size}"
    file_path = os.path.join(THRESHOLD_PATH, file_name)

    if os.path.exists(file_path):
        res = []
        with open(file_path, 'r') as file:
            for line in file:
                if line != "\n":
                    res.append((float(line)))
        return res
    else:
        return []


def run_experiment(eps, ensemble_size):
    attack_threshold = manage_threshold(eps, ensemble_size)
    exp1 = BASIC_EXPERIEMENT(eps=eps, ensemble_size=ensemble_size, attack_threshold=attack_threshold)
    identifier = exp1.generate_model_identifier(exp1.eps, exp1.ensemble_size, exp1.batch_size, "resnet18", "CIFAR10")
    if exp1.model_exists(identifier, 0):
        exp1.load_config(f"./saved_models/{identifier}_config.json")

    exp1.get_CIFAR_ten()
    print("train ", ensemble_size, "\n")

    exp1.train_models()
    print("prepare attack ", ensemble_size, "\n")

    exp1.prepare_attack()
    print("attack ", ensemble_size, "\n")
    y_average, y_max = exp1.make_experiment()

    attack_threshold = manage_threshold((eps * ensemble_size), 1)
    exp2 = BASIC_EXPERIEMENT(eps=(eps * ensemble_size), ensemble_size=1, attack_threshold=attack_threshold)
    identifier = exp2.generate_model_identifier(exp2.eps, exp2.ensemble_size, exp2.batch_size, "resnet18", "CIFAR10")
    if exp2.model_exists(identifier, 0):
        exp2.load_config(f"./saved_models/{identifier}_config.json")

    exp2.get_CIFAR_ten()
    print("train 1\n")

    exp2.train_models()
    print("prepare attack 1\n")

    exp2.prepare_attack()
    print("attack 1\n")
    y_single = exp2.make_experiment()

    plot_ROC_curve(y_average, y_max, y_single, eps, ensemble_size, exp1.number_of_samples)

def check_individual_model_results(eps, ensemble_size):
    attack_threshold = manage_threshold(eps, ensemble_size)
    exp1 = BASIC_EXPERIEMENT(eps=eps, ensemble_size=ensemble_size, attack_threshold=attack_threshold)
    identifier = exp1.generate_model_identifier(exp1.eps, exp1.ensemble_size, exp1.batch_size, "resnet18", "CIFAR10")
    if exp1.model_exists(identifier, 0):
        exp1.load_config(f"./saved_models/{identifier}_config.json")

    exp1.get_CIFAR_ten()
    print("train ", ensemble_size, "\n")

    exp1.train_models()
    print("prepare attack ", ensemble_size, "\n")

    exp1.prepare_attack_individual_model()

    print("attack")
    exp1.attack_individual_model()

    attack_threshold = manage_threshold((eps * ensemble_size), 1)
    exp2 = BASIC_EXPERIEMENT(eps=(eps * ensemble_size), ensemble_size=1, attack_threshold=attack_threshold)
    identifier = exp2.generate_model_identifier(exp2.eps, exp2.ensemble_size, exp2.batch_size, "resnet18", "CIFAR10")
    if exp2.model_exists(identifier, 0):
        exp2.load_config(f"./saved_models/{identifier}_config.json")

    exp2.get_CIFAR_ten()

    print("train 1\n")
    exp2.train_models()

    print("prepare attack 1\n")
    exp2.prepare_attack_individual_model()

    print("attack 1\n")
    exp2.attack_individual_model()




if __name__ == '__main__':
    epsilon = 8
    n = 2
    run_experiment(epsilon, n)