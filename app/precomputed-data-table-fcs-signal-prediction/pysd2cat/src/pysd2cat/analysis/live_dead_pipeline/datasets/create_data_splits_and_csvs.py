import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)

data_path = os.path.join(os.getcwd(), "datasets")


def main():
    # NOTE: The code in this module only needs to be ran once. Do not run it if you have already done so.
    # I went ahead and commented out the to_csv lines just as a precaution (even though there is a random seed).
    # I created this module because splitting takes a while to run,
    # so it saves the splits as csv's so they can be read in by model code in other modules.

    # choose organism
    organism = "bacillus"

    if organism == "yeast":
        feature_cols = ["FSC-A", "SSC-A", "BL1-A", "RL1-A", "FSC-H", "SSC-H", "BL1-H", "RL1-H", "FSC-W", "SSC-W", "BL1-W", "RL1-W"]
    elif organism in ["bacillus", "ecoli"]:
        feature_cols = ["FSC-A", "SSC-A", "RL1-A", "FSC-H", "SSC-H", "RL1-H", "FSC-W", "SSC-W", "RL1-W"]
    else:
        raise NotImplementedError("{} is not implemented. "
                                  "Please choose from yeast, bacillus, or ecoli".format(organism))

    print("\n**************************** CHOSEN ORGANISM IS: {} ****************************\n".format(organism))

    # Read in the full dataset. Each dataset is assumed to have the same structure pretty much.
    full_df = pd.read_csv(os.path.join(data_path, "full_{}_df_cleaned.csv".format(organism)))
    print(full_df.head())
    print("Shape of full_df: {}\n".format(full_df.shape))

    # The following code takes an equal number of samples from each ethanol, time_point, and stain group
    # to ensure each class and stain combo has the same number of sample points.
    # this is done to have balanced classes both when using stained samples and when using non-stained samples.
    min_num_events = full_df.groupby(["ethanol", "time_point", "stain"]).size().min()
    print("min_num_events: {}\n".format(min_num_events))
    balanced_df = full_df.groupby(["ethanol", "time_point", "stain"]). \
        apply(lambda x: x.sample(min_num_events, random_state=5)).reset_index(drop=True)
    print("Shape of balanced_df: {}".format(balanced_df.shape))

    # Create Log10 features
    logged_feature_cols = ["log_{}".format(f) for f in feature_cols]
    for f, lf in zip(feature_cols, logged_feature_cols):
        balanced_df.loc[:, lf] = list(map(lambda x: np.log10(x) if x > 0 else 0, balanced_df[f]))
    print(balanced_df)
    print()

    # Create train/test splits here. Stratify on ["stain", "ethanol", "time_point"]
    train_bank, test_df = train_test_split(balanced_df, test_size=0.3, random_state=5,
                                           stratify=balanced_df[["stain", "ethanol", "time_point"]])

    # Normalize features: make sure to do it separately for stain=0 and stain=1 rows.
    normalized_train_bank = train_bank.copy()
    normalized_test_df = test_df.copy()
    scaler_0 = StandardScaler()
    scaler_1 = StandardScaler()

    scaler_0.fit(normalized_train_bank.loc[normalized_train_bank["stain"] == 0, feature_cols + logged_feature_cols])
    normalized_train_bank.loc[normalized_train_bank["stain"] == 0, feature_cols + logged_feature_cols] = \
        scaler_0.transform(normalized_train_bank.loc[normalized_train_bank["stain"] == 0, feature_cols + logged_feature_cols])
    normalized_test_df.loc[normalized_test_df["stain"] == 0, feature_cols + logged_feature_cols] = \
        scaler_0.transform(normalized_test_df.loc[normalized_test_df["stain"] == 0, feature_cols + logged_feature_cols])

    scaler_1.fit(normalized_train_bank.loc[normalized_train_bank["stain"] == 1, feature_cols + logged_feature_cols])
    normalized_train_bank.loc[normalized_train_bank["stain"] == 1, feature_cols + logged_feature_cols] = \
        scaler_1.transform(normalized_train_bank.loc[normalized_train_bank["stain"] == 1, feature_cols + logged_feature_cols])
    normalized_test_df.loc[normalized_test_df["stain"] == 1, feature_cols + logged_feature_cols] = \
        scaler_1.transform(normalized_test_df.loc[normalized_test_df["stain"] == 1, feature_cols + logged_feature_cols])

    print("Shape of normalized_train_bank: {}".format(normalized_train_bank.shape))
    print("Shape of normalized_test_df: {}\n".format(normalized_test_df.shape))
    print(normalized_test_df)
    print()

    # Output csvs
    # train_bank.to_csv(os.path.join(data_path, "{}_train_bank.csv".format(organism)), index=False)
    # test_df.to_csv(os.path.join(data_path, "{}_test_df.csv".format(organism)), index=False)
    # normalized_train_bank.to_csv(os.path.join(data_path, "{}_normalized_train_bank.csv".format(organism)), index=False)
    # normalized_test_df.to_csv(os.path.join(data_path, "{}_normalized_test_df.csv".format(organism)), index=False)


if __name__ == '__main__':
    main()
