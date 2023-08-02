import torch

from training import train_model

class LiRA:
    def __init__(self, eps, data, sample_index, model, N=5000, batch_size=64):
        """
        :param eps: epsilon DP of the shadow models.
        :param data:
        :param sample:
        :param N: number of shadow models.
        """
        self.eps = eps
        self.data = data
        self.sample_index = sample_index
        self.sample, self.label = data.dataset[sample_index]
        self.model = model
        self.N = N
        self.batch_size = batch_size
        self.conf_in = []
        self.confs_out = []

        total_size = len(self.data)
        self.a_size = total_size // 2
        self.b_size = total_size - self.a_size
        self.clean_sample_from_dataframe()

    def clean_sample_from_dataframe(self):
        # Remove the sample from the dataset
        self.data.samples.pop(self.sample_index)
        self.data.targets.pop(self.sample_index)

        # Create a new DataLoader with the modified dataset
        new_dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=self.data.shuffle,
                                                     num_workers=self.data.num_workers)
        self.data = new_dataloader

    def get_in_out_training_data(self):
        # Split the dataset into half
        a_set, b_set = torch.utils.data.random_split(self.data, [self.a_size, self.b_size])

        # Create DataLoaders for training and testing data
        return torch.utils.data.DataLoader(a_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def add_sample_to_dataloader(self, dataloader):
        # Get the dataset from the DataLoader
        dataset = dataloader.dataset

        # Append the new sample and target to the dataset
        dataset.samples.append(self.sample)
        dataset.targets.append(self.label)

        # Create a new DataLoader with the updated dataset
        return torch.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=dataloader.shuffle,
                                     num_workers=dataloader.num_workers)

    def attack(self):
        for i in range(self.N):
            # get training data
            out_data = self.get_in_out_training_data()
            in_data = self.add_sample_to_dataloader(out_data)

            # train in/out shadow models
            in_model = train_model(self.eps, in_data, in_data)
            out_model = train_model(self.eps, out_data, out_data)

            # calculate confidence
            