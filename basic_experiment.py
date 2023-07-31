import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from training import train_model
from label_attack import attack

NUMBER_OF_SAMPLES = 1000

def train_models(epsilon, n, train_data, test_data):
    """
    Train n models to be epsilon-DP, using the same training set for all models.
    :param epsilon:
    :return:
    """
    models = []
    for i in range(n):
        acc, model = train_model(epsilon, train_data, test_data)
        models.append(model)
        print("model", i + 1, "accuracy: ", acc)
    models = [train_model(epsilon, train_data, test_data) for i in range(n)]
    return models

def check_average(models, sample, data):
    """
    This experiment check the basic attack when take the average of the attack of the models.
    :return:
    """
    res = [attack(model, sample, data) for model in models]
    return np.mean(res)

def check_max(models, sample, data):
    res = [attack(model, sample, data) for model in models]
    return np.max(res)

def get_training_set(data_frame, target_column, test_size=0.5, random_state=None):
    """
    Split 50% of the data to be training data, and 50% to be test data.
    :return: return train_data, test_data.
    """
    # Separate the features (input) and target (output) variables
    features = data_frame.drop(columns=[target_column])
    target = data_frame[target_column]

    # Split the data into training and test sets
    train_data, test_data, train_target, test_target = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    return train_data, test_data, train_target, test_target


def plot_ROC_curve(fpr, tpr, evaluate_score):
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
    plt.savefig("/tmp/fprtpr " + evaluate_score + ".png")
    plt.show()


def make_experiment(models, train_data, test_data, evaluate_score):
    """
    The function make the experiment and produce ROC curve.
    :param models:
    :param train_data:
    :param test_data:
    :return:
    """
    eval = {"average": check_average, "max": check_max}
    sampled_indices = np.random.choice(train_data.index, size=NUMBER_OF_SAMPLES, replace=False)
    true_sample = train_data.loc[sampled_indices]
    sampled_indices = np.random.choice(test_data.index, size=NUMBER_OF_SAMPLES, replace=False)
    false_sample = test_data.loc[sampled_indices]
    y_true = ([1] * NUMBER_OF_SAMPLES) + ([0] * NUMBER_OF_SAMPLES)
    y_score = []
    for i in range(NUMBER_OF_SAMPLES):
        y_score.append(eval[evaluate_score](models, true_sample[i], train_data))

    for i in range(NUMBER_OF_SAMPLES):
        y_score.append(eval[evaluate_score](models, false_sample[i], train_data))

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    plot_ROC_curve(fpr, tpr, evaluate_score)

