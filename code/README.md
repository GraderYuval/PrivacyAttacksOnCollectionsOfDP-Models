**Files Description**

requirements.txt - A requirements file for running the experiment.
training.py - Train ResNet18 model with (epsilon,delta)-DP.
experiment.py - Run experiment for a given size of ensemble and an epsilon for DP.
plot_results.py - Plot the results of given experiments as ROC curve.

**Environment**
To run the experiment, one need first to install requirements.txt by running _pip install -r requirements.txt_.

**Run Experiment**
To run experiment one need to call "run_experiment" from experiment.py, with the appropriate ensemble size and epsilon.

**Plot**
To plot the results of experiments one need to call "plot_results" from plot_results.py,
when the experiments are in the format:

experiments = [
        {"epsilon": 2, "ensemble": 3},
        {"epsilon": 4, "ensemble": 3},
        {"epsilon": 8, "ensemble": 3}
    ]

A list of experiments where each experiment is dictionary with the keys "epsilon" and "ensemble", represent the epsilon
for the epsilon-DP and the ensemble size.