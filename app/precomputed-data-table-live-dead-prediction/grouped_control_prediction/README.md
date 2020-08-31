# Grouped Control Prediction

The `grouped_control_prediction` package is used for running a modified version of the random forest live/dead classifier model. The name comes from the fact that we are combining, or "grouping", different control experiments to form the training set for our model. A detailed run through of the methods and results can be found in the `GroupedControlPredictionMemo.pdf` file located in the `precomputed-data-table` directory.

## Installation

Installing `grouped_control_prediction` requires the `setuptools` Python package. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install `setuptools`:

```bash
pip install setuptools
```

Next, navigate to the `grouped_control_prediction` directory and run the following command:

```bash
python setup.py install
```

Note: You may need to add `--user` at the end of this command if you are on a TACC HPC such as Wrangler.


## Usage

The three primary files are `predict.py`, `main.py`, and `eval.py`.

`predict.py` simply runs one iteration of the random forest classifier from the `pysd2cat` package. It returns the predicted live/dead labels, the training set used in the classifier, the average Wasserstein distance of the controls in the training set with respect to the prediction set, and the test accuracy of the model.

`main.py` calls `predict.py` to run a single classification iteration and also outputs well-based time series plots and overlaid prediction vs training set histograms. This file is used in the Jupyter notebook `demo_grouped_control_prediction` to compare the random forest predictions to the optical density predictions.

`eval.py` is similar to `main.py` except that it does not output any plots and performs all of the optical density comparisons within itself. It returns the metrics `od_loss` and `od_accuracy`, whose calculations are explained in the Memo. This file is used primarily to evaluate the performance of the random forest classifier over different training set sizes by varying the number of control sets added, with performance measured by test accuracy, od_loss, and od_accuracy. These evaluations can be found in the notebook `ControlSizeEvaluation`.

## Utils

The `utils` folder contains the files `data_utils.py` and `plot.py` that help with loading data and outputting plots respectively.

## Package Resources

The `data` folder contains important resources for running the classifier. It holds the Wasserstein distances of various prediction datasets with respect to the set of control experiments used to construct the training set. These distances are stored as pickled data files and are read in by `predict.py` when deciding which controls to add to the training set.