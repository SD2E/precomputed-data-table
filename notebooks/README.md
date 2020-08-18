# Grouped Control Prediction Notebooks

The following is an explanation of the various Jupyter notebooks and data files located in the `notebooks` directory.

## demo_grouped_control_prediction.ipynb

This notebook is meant to parallel `demo_fcs_signal_prediction.ipynb`. It runs a single iteration of the random forest classifier using the `pysd2cat` package, outputting the predicted labels on the inputted prediction set. It also outputs time series plots of the predictions based on well along with overlaid prediction vs training set histograms.

The outputted live/dead labels from the random forest classifier are then merged with the optical density data from the same prediction set and saved. This merged dataframe containing both sets of predictions will then be inputted into the `rf_od_comparison.ipynb` notebook to be analyzed further.

## rf_od_comparison.ipynb

This notebook takes in a merged dataframe containing both the random forest predictions and the optical density predictions with respect to a single prediction experiment. Some exploratory data analysis is performed on the data before fitting a logistic regression onto both sets of predictions. The random forest labels are used as the independent variables, and the optical density labels are used as the dependent variables.

## ControlSizeEvaluation.ipynb

This notebook is used for evaluating the grouped control technique for assembling our model's training set. We iterate over the values [5,10,15,20,25] for the control size variable, testing for the optimal number of control experiments to build our training set with. Each classification run for each size is also replicated five times to account for the randomization in the train/test split and of the random forest classifier itself.

The output of the 25 runs is saved as a single pickled dataframe in the `notebook_data/sample_size_eval/weighted/` directory. Finally, the output is plotted against three performance metrics: test accuracy, mean optical density loss, and optical density accuracy.

## K-NearestControls.ipynb

This notebook is used to compute the Wasserstein distances between the inputted prediction experiment and each control experiment in the given set of experiments that contains ground truth labels. These distances are then saved in `notebook_data/wasserstein_distances/` as pickled Python dictionaries.

## AggregateExperimentResults.ipynb

This notebook takes all of the merged random forest and optical density prediction dataframes and combines them into a single dataframe. This aggregate dataframe is then split into separate match and mismatch dataframes. Matches refer to observations that received the same label from both techniques, while mismatches are observations with conflicting labels. These two dataframes are stored in the directory `notebook_data/misc/`.

## Cells_mL_Analysis.ipynb

This notebook explores the `cells/mL` column of the mismatched portion of the `YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208_20200610192131` experiment dataset. It attempts to identify any effect that variable has on the predicted labels.

## notebook_data

### misc

Contains matched and mismatched datasets from all prediction sets used. Includes `cells/mL` analysis dataset and a folder called `growth_data` containing updated optical density data.

### predictions

Contains the merged random forest and optical density predictions datasets saved as CSV files.

### sample_size_eval

Stores the output from the control size evaluation notebook.

### wasserstein_distances

Stores the Wasserstein distances between each prediction set and the given set of controls with ground truth labels.