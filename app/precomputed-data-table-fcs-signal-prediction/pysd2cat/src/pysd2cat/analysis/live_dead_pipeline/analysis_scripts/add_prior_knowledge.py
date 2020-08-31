import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from scipy.optimize import curve_fit

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def unscaled_gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def true_gaussian(x, mu, sig):
    return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


def main():
    # import data and initialize some vars
    yeast_train_bank = pd.read_csv("datasets/yeast_normalized_train_bank.csv")
    yeast_test_df = pd.read_csv("datasets/yeast_normalized_test_df.csv")
    print("Shape of yeast_train_bank: {}".format(yeast_train_bank.shape))
    print("Shape of yeast_test_df: {}".format(yeast_test_df.shape))
    print()
    yeast_features_0 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A"]
    yeast_features_1 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A", "RL1-A", "RL1-W", "RL1-H"]

    # set stain here. 0 = don't use stains, 1 = use stains
    stain = 1
    if stain == 0:
        print("Unstained samples will be used...")
        fcols_yeast = yeast_features_0
        yeast_train_bank = yeast_train_bank.loc[yeast_train_bank["stain"] == 0]
        yeast_test_df = yeast_test_df.loc[yeast_test_df["stain"] == 0]
    elif stain == 1:
        print("Stained samples will be used...")
        fcols_yeast = yeast_features_1
        yeast_train_bank = yeast_train_bank.loc[yeast_train_bank["stain"] == 1]
        yeast_test_df = yeast_test_df.loc[yeast_test_df["stain"] == 1]
    else:
        raise NotImplementedError()
    print("Shape of yeast_train_bank: {}".format(yeast_train_bank.shape))
    print("Shape of yeast_test_df: {}".format(yeast_test_df.shape))
    print()

    # subsample yeast_train_bank to create a smaller df called yeast_train_df
    yeast_train_df, _ = train_test_split(yeast_train_bank, train_size=0.1, random_state=5,
                                         stratify=yeast_train_bank[["ethanol", "time_point"]])
    yeast_test_df, _ = train_test_split(yeast_test_df, train_size=0.1, random_state=5,
                                        stratify=yeast_test_df[["ethanol", "time_point"]])
    print("Shape of yeast_train_df: {}".format(yeast_train_df.shape))
    print("Shape of yeast_test_df: {}".format(yeast_test_df.shape))
    print()

    # Set scatter col
    scatter_col = "log_FSC-A"
    scatter_vals = yeast_train_df[scatter_col]
    scatter_mean = scatter_vals.mean()
    scatter_std = np.std(scatter_vals)
    # print(scatter_mean)
    # print(scatter_std)
    gaussian_vals = unscaled_gaussian(x=scatter_vals, mu=scatter_mean, sig=scatter_std)
    # plt.scatter(x=scatter_vals, y=gaussian_vals)
    # plt.show()

    # Histogram of scatter:
    # plt.hist(scatter_vals, bins=100)
    # plt.axvline(scatter_vals.mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.show()
    # sys.exit(0)

    # Half-Gaussian weighting of samples
    gaussian_vals.rename("gaussian_vals", inplace=True)
    weight_df = pd.concat([scatter_vals, gaussian_vals], axis=1)
    weight_df["sample_weight"] = gaussian_vals
    weight_df.loc[weight_df[scatter_col] > scatter_mean, "sample_weight"] = 1
    # print(weight_df)

    # Create Models:
    print("\nModel creation time...\n")

    # baseline 1
    print("\nRunning Baseline (class_weight=balanced)")
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight="balanced")
    rfc.fit(X=yeast_train_df[fcols_yeast], y=yeast_train_df["ethanol"], sample_weight=None)
    preds = rfc.predict(X=yeast_test_df[fcols_yeast])
    acc = accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
    bacc = balanced_accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
    cr = classification_report(y_true=yeast_test_df["ethanol"], y_pred=preds)
    print(acc)
    print(bacc)
    print(cr)

    # baseline 2
    print("\nRunning Baseline (class_weight=None)")
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight=None)
    rfc.fit(X=yeast_train_df[fcols_yeast], y=yeast_train_df["ethanol"], sample_weight=None)
    preds = rfc.predict(X=yeast_test_df[fcols_yeast])
    acc = accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
    bacc = balanced_accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
    cr = classification_report(y_true=yeast_test_df["ethanol"], y_pred=preds)
    print(acc)
    print(bacc)
    print(cr)
    print()

    # removing samples method
    r_cut = -1.5
    print(yeast_train_df["ethanol"].value_counts())
    print()
    train_removed = yeast_train_df.loc[yeast_train_df[scatter_col] >= r_cut]
    test_removed = yeast_test_df.loc[yeast_test_df[scatter_col] >= r_cut]
    print(train_removed["ethanol"].value_counts())
    print()
    print("\nRunning Removing Samples Method (f-scatter)")
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight=None)
    rfc.fit(X=train_removed[fcols_yeast], y=train_removed["ethanol"], sample_weight=None)
    preds = rfc.predict(X=test_removed[fcols_yeast])
    acc = accuracy_score(y_true=test_removed["ethanol"], y_pred=preds)
    bacc = balanced_accuracy_score(y_true=test_removed["ethanol"], y_pred=preds)
    cr = classification_report(y_true=test_removed["ethanol"], y_pred=preds)
    print(acc)
    print(bacc)
    print(cr)

    # weighting samples method
    dict_of_sample_weights = {"Half-Uniform": np.array([0.01 if x < scatter_mean else 1 for x in yeast_train_df[scatter_col]]),
                              "Half-Gaussian": list(weight_df["sample_weight"])}

    for key, sample_weights in dict_of_sample_weights.items():
        print("\nRunning Weighting Samples Method by {}: {}".format(scatter_col, key))
        rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                     n_jobs=-1, class_weight=None)
        rfc.fit(X=yeast_train_df[fcols_yeast], y=yeast_train_df["ethanol"], sample_weight=sample_weights)
        preds = rfc.predict(X=yeast_test_df[fcols_yeast])
        acc = accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
        bacc = balanced_accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
        cr = classification_report(y_true=yeast_test_df["ethanol"], y_pred=preds)
        print(acc)
        print(bacc)
        print(cr)

    # Removing time-points method:
    time_col = "time_point"
    print(yeast_train_df["ethanol"].value_counts())
    train_removed = yeast_train_df.loc[yeast_train_df[time_col] == 12]
    test_removed = yeast_test_df.loc[yeast_test_df[time_col] == 12]
    print(train_removed["ethanol"].value_counts())
    print()
    print("\nRunning Removing Samples Method (time_point)")
    rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                 n_jobs=-1, class_weight=None)
    rfc.fit(X=train_removed[fcols_yeast], y=train_removed["ethanol"], sample_weight=None)
    preds = rfc.predict(X=test_removed[fcols_yeast])
    acc = accuracy_score(y_true=test_removed["ethanol"], y_pred=preds)
    bacc = balanced_accuracy_score(y_true=test_removed["ethanol"], y_pred=preds)
    cr = classification_report(y_true=test_removed["ethanol"], y_pred=preds)
    print(acc)
    print(bacc)
    print(cr)

    # weighting samples according to time_point:
    dict_of_sample_weights = {"Half-Uniform": np.array([0.2 if x < 12 else 1 for x in yeast_train_df[time_col]])}

    for key, sample_weights in dict_of_sample_weights.items():
        print("\nRunning Weighting Samples Method by time_point: {}".format(key))
        rfc = RandomForestClassifier(n_estimators=361, max_features='auto', criterion='entropy', min_samples_leaf=13,
                                     n_jobs=-1, class_weight=None)
        rfc.fit(X=yeast_train_df[fcols_yeast], y=yeast_train_df["ethanol"], sample_weight=sample_weights)
        preds = rfc.predict(X=yeast_test_df[fcols_yeast])
        acc = accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
        bacc = balanced_accuracy_score(y_true=yeast_test_df["ethanol"], y_pred=preds)
        cr = classification_report(y_true=yeast_test_df["ethanol"], y_pred=preds)
        print(acc)
        print(bacc)
        print(cr)


if __name__ == '__main__':
    main()
