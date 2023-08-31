import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc

THRESHOLD_PATH = "threshold/"
RESULTS_PATH = "results/"


def load_results(result_name):
    with open(f"results/{result_name}", 'r') as file:
        data = file.readlines()
    return np.array([float(d.strip()) for d in data])


def plot_ROC_curves(results, epsilons, ensembles, number_of_samples):
    sns.set_style("whitegrid")  # Setting Seaborn style
    plt.figure(figsize=(10, 8))
    y_true = ([1] * number_of_samples) + ([0] * number_of_samples)

    # Define Line styles
    linestyles = ['-', '--', ':']

    # Using a different color for each experiment
    palette = sns.color_palette("husl", len(results))

    for i, (y_average, y_max, y_single) in enumerate(results):
        # plot average roc curve
        fpr_a, tpr_a, _ = roc_curve(y_true, y_average, pos_label=1)
        roc_auc_a = auc(fpr_a, tpr_a)
        sns.lineplot(fpr_a, tpr_a, label=f'Eps-{epsilons[i]} Ens-{ensembles[i]} Average (AUC = {roc_auc_a:.3f})',
                     color=palette[i], linestyle=linestyles[0])

        # plot max roc curve
        fpr_m, tpr_m, _ = roc_curve(y_true, y_max, pos_label=1)
        roc_auc_m = auc(fpr_m, tpr_m)
        sns.lineplot(fpr_m, tpr_m, label=f'Eps-{epsilons[i]} Ens-{ensembles[i]} Max (AUC = {roc_auc_m:.3f})',
                     color=palette[i], linestyle=linestyles[1])

        # plot single model roc curve
        fpr_s, tpr_s, _ = roc_curve(y_true, y_single, pos_label=1)
        roc_auc_s = auc(fpr_s, tpr_s)
        sns.lineplot(fpr_s, tpr_s, label=f'Eps-{epsilons[i] * ensembles[i]} Ens-1 Single (AUC = {roc_auc_s:.3f})',
                     color=palette[i], linestyle=linestyles[2])

    # Common ROC plot properties
    plt.plot([0, 1], [0, 1], color='gray', linestyle='-.')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title("ROC Curves")

    # Adjust the legend
    leg = plt.legend(loc='lower right', fontsize='large')
    for i in range(0, len(leg.get_lines()), 3):
        leg.get_lines()[i].set_linestyle("-")
        leg.get_lines()[i + 1].set_linestyle("--")
        leg.get_lines()[i + 2].set_linestyle(":")

    plt.tight_layout()  # Making sure everything fits nicely
    plt.savefig("combined_roc_curve.png")
    plt.show()


if __name__ == '__main__':
    # Just add or remove experiment settings from this list to handle any number of experiments
    experiments = [
        # {"epsilon": 2, "ensemble": 2},
        {"epsilon": 2, "ensemble": 3},
        {"epsilon": 2, "ensemble": 4},
        # {"epsilon": 8, "ensemble": 3},
        # {"epsilon": 4, "ensemble": 3},
        # {"epsilon": 8, "ensemble": 3},
    ]

    results = []
    epsilons = []
    ensembles = []

    for experiment in experiments:
        y_avg = load_results(f"results_eps_{experiment['epsilon']}_ensemble_{experiment['ensemble']}_average")
        y_max = load_results(f"results_eps_{experiment['epsilon']}_ensemble_{experiment['ensemble']}_max")
        y_single = load_results(f"results_eps_{experiment['epsilon'] * experiment['ensemble']}_ensemble_1_max")
        results.append((y_avg, y_max, y_single))
        epsilons.append(experiment['epsilon'])
        ensembles.append(experiment['ensemble'])

    plot_ROC_curves(results, epsilons, ensembles, 1000)
