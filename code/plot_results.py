import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

LINE_STYLE = ['-', '--', ':']
COLORS = ["red", "green", "blue"]


def plot_ROC_curve(results, epsilons, ensembles, number_of_samples):
    """
    Calculate TPR, FPR and AUC for each part of each experiment, and plot all results of all experiments.
    """
    plt.figure(figsize=(8, 6))
    y_true = ([1] * number_of_samples) + ([0] * number_of_samples)

    for i, (y_average, y_max, y_single) in enumerate(results):
        # plot average roc curve
        fpr_a, tpr_a, thresholds_a = roc_curve(y_true, y_average, pos_label=1)
        roc_auc_a = auc(fpr_a, tpr_a)
        plt.plot(fpr_a, tpr_a, color=COLORS[i % 3], lw=2,
                 label=f'Eps-{epsilons[i]} Ens-{ensembles[i]} Average (AUC = {roc_auc_a:.3f})', linestyle=LINE_STYLE[0])

        # plot max roc curve
        fpr_m, tpr_m, thresholds_m = roc_curve(y_true, y_max, pos_label=1)
        roc_auc_m = auc(fpr_m, tpr_m)
        plt.plot(fpr_m, tpr_m, color=COLORS[i % 3], lw=2,
                 label=f'Eps-{epsilons[i]} Ens-{ensembles[i]} Max (AUC = {roc_auc_m:.3f})', linestyle=LINE_STYLE[1])

        # plot single model roc curve
        fpr_s, tpr_s, thresholds_s = roc_curve(y_true, y_single, pos_label=1)
        roc_auc_s = auc(fpr_s, tpr_s)
        plt.plot(fpr_s, tpr_s, color=COLORS[i % 3], lw=2,
                 label=f'Eps-{epsilons[i] * ensembles[i]} Ens-1 Single (AUC = {roc_auc_s:.3f})',
                 linestyle=LINE_STYLE[2])

    # Plot the ROC curve
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
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

def load_results(result_name):
    """
    Load the results of the experiment from file to an array.
    """
    with open(f"results/{result_name}", 'r') as file:
        data = file.readlines()
    return np.array([float(d.strip()) for d in data])

def manage_experiments(experiments):
    """
    Load the results of all experiments to arrays, and create lists suit to plot_ROC_curve function.
    """
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

    return results, epsilons, ensembles

def plot_results(experiments):
    """
    Plot the results of a given list of experiments.
    """
    results, epsilons, ensembles = manage_experiments(experiments)
    plot_ROC_curve(results, epsilons, ensembles, number_of_samples=1000)