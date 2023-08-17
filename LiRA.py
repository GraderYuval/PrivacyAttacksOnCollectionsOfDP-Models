import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
import copy

from training import train_model

class LiRA:
    def __init__(self, eps, dataset, sample_index, device, N=16, batch_size=64):
        """
        :param eps: epsilon DP of the shadow models.
        :param data:
        :param sample:
        :param N: number of shadow models.
        """
        self.eps = eps
        self.dataset = dataset
        self.sample_index = sample_index
        self.sample, self.label = dataset.data[sample_index], dataset.targets[sample_index]
        # change sample to be in shape to be input of a model
        self.sample = np.array([self.sample])
        self.sample = np.append(self.sample, self.sample, axis=0)

        self.device = device
        self.N = N
        self.batch_size = batch_size
        self.confs_in = []
        self.confs_out = []
        self.clean_sample_from_dataframe()

    def clean_sample_from_dataframe(self):
        """
        Delete sample from dataset.
        :return:
        """
        # Remove the sample from the dataset
        self.dataset.data = np.delete(self.dataset.data, self.sample_index, axis=0)
        del self.dataset.targets[self.sample_index]

    def get_in_out_training_data(self):
        """
        Sample training set, and return it as out_data which is the training set without the sample, and in data which
        is the training set with the sample.
        :return:
        """
        # Split the dataset into half
        # Calculate the size of the dataset
        total_size = len(self.dataset)
        a_size = total_size // 2
        b_size = total_size - a_size

        # Split the dataset into training and testing sets
        data_a, data_b = \
            torch.utils.data.random_split(self.dataset, [a_size, b_size])
        out_data = torch.utils.data.DataLoader(data_a, batch_size=self.batch_size, shuffle=True)
        data_b.indices = copy.deepcopy(data_a.indices)
        data_b.indices.append(self.sample_index)
        in_data = torch.utils.data.DataLoader(data_b, batch_size=self.batch_size, shuffle=True)
        return in_data, out_data

    def calc_confidence(self, in_model, out_model):
        """
        The function calculate the confidence the in model and out model has for the sample.
        :param in_model:
        :param out_model:
        :return:
        """
        in_model.eval()
        out_model.eval()
        sample_tensor = torch.tensor(self.sample, dtype=torch.float32)
        sample_tensor = torch.permute(sample_tensor, (0, 3, 1, 2))
        sample_tensor = sample_tensor.to('cpu')
        with torch.no_grad():
            # not sure if it's true
            in_p = in_model(sample_tensor)
            out_p = out_model(sample_tensor)
        in_p = F.softmax(in_p, dim=1)
        out_p = F.softmax(out_p, dim=1)
        in_p = in_p[0][self.label].item()
        out_p = out_p[0][self.label].item()
        # need to think what to do when in_p or out_p equal to 1
        if in_p == 1:
            self.confs_in.append(36.7368005696771)
        elif in_p == 0:
            self.confs_in.append(-36.841361487904734)
        else:
            self.confs_in.append(np.log(in_p / (1 - in_p)))
        if out_p == 1:
            self.confs_out.append(36.7368005696771)
        elif out_p == 0:
            self.confs_out.append(-36.841361487904734)
        else:
            self.confs_out.append(np.log(out_p / (1 - out_p)))

    def prepare_attack(self):
        for i in range(self.N):
            # get training data
            in_data, out_data = self.get_in_out_training_data()

            # train in/out shadow models
            torch.cuda.empty_cache()
            in_model = train_model(self.eps, in_data, self.device)
            in_model = in_model.to('cpu')

            out_model = train_model(self.eps, out_data, self.device)
            out_model = out_model.to('cpu')

            # calculate confidence
            self.calc_confidence(in_model, out_model)
            del in_model
            del out_model

    def attack(self, model):
        """
        The function implement LiRA attack.
        :return:
        """
        torch.cuda.empty_cache()
        # calculate expectation and variance of the confidence of the sample, for the in models and out models.
        mean_in = np.mean(self.confs_in)
        mean_out = np.mean(self.confs_out)
        var_in = np.var(self.confs_in)
        var_out = np.var(self.confs_out)

        # calculate confidence of the sample of the attacked model
        sample_tensor = torch.tensor(self.sample, dtype=torch.float32)
        sample_tensor = torch.permute(sample_tensor, (0, 3, 1, 2))
        sample_tensor = sample_tensor.to('cpu')
        model = model.to('cpu')
        model_confs = model(sample_tensor)
        model_confs = F.softmax(model_confs, dim=1)
        model_confs = model_confs[0][self.label].item()
        if model_confs == 1:
            model_confs = 36.7368005696771
        elif model_confs == 0:
            model_confs = -36.841361487904734
        else:
            model_confs = np.log(model_confs / (1 - model_confs))
        torch.cuda.empty_cache()

        # return likelihood ratio test
        return norm.pdf(model_confs, mean_in, var_in) / (norm.pdf(model_confs, mean_out, var_out))