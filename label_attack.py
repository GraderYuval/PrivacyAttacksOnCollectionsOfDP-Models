import torch
from torch import nn
import numpy as np
from art.utils import load_mnist
import torch.optim as optim
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from numpy.random import choice
from sklearn.metrics import accuracy_score


class MembershipInfernceAttack():
    def __init__(self, model, device, distance_threshold=None):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters())
        self.device = device

        self.model.to(self.device)
        self.distance_threshold_tau=distance_threshold
        self.mia_label_only = None

    def fit(self):
        ''' Loading MNIST dataset '''
        print("Loading MNIST dataset")
        (x_train, y_train), (x_test, y_test), _min, _max = load_mnist(raw=True)

        x_train = np.expand_dims(x_train, axis=1).astype(np.float32)
        x_test = np.expand_dims(x_test, axis=1).astype(np.float32)

        
        ''' Fitting the given model (self.model) on the data and evaluating its accuracy '''
        art_model = PyTorchClassifier(model=self.model, loss=self.criterion, optimizer=self.optimizer, channels_first=True, input_shape=(1,28,28,), nb_classes=10, clip_values=(_min,_max))
        art_model.fit(x_train, y_train, nb_epochs=10, batch_size=128)
        pred = np.array([np.argmax(arr) for arr in art_model.predict(x_test)])
        print('Base model accuracy: ', np.sum(pred == y_test) / len(y_test))


        ''' 
            Initializing the attack module. 
            If a given threshold is given, skipping to the membership inference evaluations.
            Else, fitting the attack with respect to the model and the data. This section takes 1-2 hours on my machine... (NADAV)
        '''
        print("Calibrating distnce threshold")
        self.mia_label_only = LabelOnlyDecisionBoundary(art_model, distance_threshold_tau=self.distance_threshold_tau)
        if self.mia_label_only.distance_threshold_tau is None:
            # number of samples used to calibrate distance threshold
            attack_train_size = 1500
            attack_test_size = 1500
            self.mia_label_only.calibrate_distance_threshold(x_train[:attack_train_size], y_train[:attack_train_size],
                                                        x_test[:attack_test_size], y_test[:attack_test_size])


        ''' Calculating the attack success on the same data set '''
        # evaluation data

        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        training_sample = np.array([1] * len(x_train) + [0] * len(x_test))

        n = 500
        eval_data_idx = choice(len(x), n)
        x_eval, y_eval = x[eval_data_idx], y[eval_data_idx]
        eval_label = training_sample[eval_data_idx]

        pred_label = self.mia_label_only.infer(x_eval, y_eval)
        print("Accuracy: %f" % accuracy_score(eval_label, pred_label))


    def attack(self, sample : np.ndarray):
        """
        Basic membership inference attack. Not sure if the data is necessary.
        :param sample:
        :return:
        """
        pred_label = self.mia_label_only.infer(sample)
        return pred_label


if __name__ == "__main__":

    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
        )
    
    distance_threshold = 123.91295623779297
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    mia = MembershipInfernceAttack(model, distance_threshold=distance_threshold, device=device)
    mia.fit()
    aa=2