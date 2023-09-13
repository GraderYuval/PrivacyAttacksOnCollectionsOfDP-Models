
# Privacy Attacks Against Collections of Differentially Private (DP) Models

### Files

requirements.txt: A requirements file for running the experiment.\
training.py: Train ResNet18 model with (epsilon,delta)-DP.\
experiment.py: Run experiment for a given size of ensemble and an epsilon for DP.\
plot_results.py: Plot the results of given experiments as ROC curve.


### Environment

To run the experiment, you need to first install the dependencies in requirements.txt by running the following command:
```
pip install -r requirements.txt
```

### Run Experiment

To run the experiment, you need to call the ```run_experiment()``` function from ```experiment.py```. The function takes two arguments: the ensemble size and the epsilon. 

### Plot Results

To plot the results of the experiments, you need to call the ```plot_results()``` function from ```plot_results.py```. The function takes one argument: a list of experiments. Each experiment is a dictionary with the keys epsilon and ensemble, for example:

```
experiments = [
    {
        "epsilon": 2,
        "ensemble": 3,
    },
    {
        "epsilon": 4,
        "ensemble": 3,
    },
    {
        "epsilon": 8,
        "ensemble": 3,
    },
]
```
### Example Experiments

The following are some example experiments that you can run:

An experiment with an ensemble size of 3 and an epsilon of 2.\
An experiment with an ensemble size of 5 and an epsilon of 4.\
An experiment with an ensemble size of 10 and an epsilon of 8.\
You can run these experiments by passing the appropriate arguments to the `run_experiment()` function.