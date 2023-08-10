import numpy as np
import torch
import torch.nn.functional as F

from training import train_model

class LiRA:
    def __init__(self, eps, dataset, sample_index, model, device, N=5000, batch_size=64):
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
        self.model = model
        self.device = device
        self.N = N
        self.batch_size = batch_size
        self.confs_in = []
        self.confs_out = []

        total_size = len(self.dataset)
        self.a_size = total_size // 2
        self.b_size = total_size - self.a_size
        self.clean_sample_from_dataframe()

    def clean_sample_from_dataframe(self):
        # Remove the sample from the dataset
        self.dataset.data = np.delete(self.dataset.data, self.sample_index, axis=0)
        del self.dataset.targets[self.sample_index]

    def get_in_out_training_data(self):
        # Split the dataset into half
        a_set, b_set = torch.utils.data.random_split(self.dataset, [self.a_size, self.b_size])
        out_data = torch.utils.data.DataLoader(a_set, batch_size=self.batch_size, shuffle=True)
        a_set.data = np.append(a_set, self.sample, axis=0)
        a_set.targets.append(self.label)
        in_data = torch.utils.data.DataLoader(a_set, batch_size=self.batch_size, shuffle=True)
        return in_data, out_data

    def calc_confidence(self, in_model, out_model):
        in_model.eval()
        out_model.eval()
        sample_tensor = torch.tensor(self.sample, dtype=torch.float32)
        with torch.no_grad():
            # not sure if it's true
            in_p = in_model(sample_tensor.unsqueeze(0))
            out_p = out_model(sample_tensor.unsqueeze(0))
        in_p = F.softmax(in_p, dim=1)
        out_p = F.softmax(out_p, dim=1)
        in_p = in_p[0][self.label].item()
        out_p = out_p[0][self.label].item()
        # need to think what to do when in_p or out_p equal to 1
        if in_p != 1:
            self.confs_in.append(np.log(in_p / (1 - in_p)))
        if out_p != 1:
            self.confs_out.append(np.log(out_p / (1 - out_p)))

    def attack(self):
        for i in range(self.N):
            # get training data
            in_data, out_data = self.get_in_out_training_data()

            # train in/out shadow models
            in_model = train_model(self.eps, in_data, self.device)
            out_model = train_model(self.eps, out_data, self.device)

            # calculate confidence
            self.calc_confidence(in_model, out_model)
        mean_in = np.mean(self.confs_in)
        mean_out = np.mean(self.confs_out)
        var_in = np.var(self.confs_in)
        var_out = np.var(self.confs_out)
        sample_tensor = torch.tensor(self.sample, dtype=torch.float32)
        model_confs = self.model(sample_tensor.unsqueeze(0))
        model_confs = F.softmax(model_confs, dim=1)
        model_confs = model_confs[0][self.label].item()
        model_confs = np.log(model_confs / (1 - model_confs))
        # return likelihood ratio test
