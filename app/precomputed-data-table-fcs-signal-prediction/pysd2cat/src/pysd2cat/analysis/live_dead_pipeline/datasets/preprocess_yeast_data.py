import os
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    # Read in the yeast dataset
    df_yeast = pd.read_csv("full_yeast_df_semi_cleaned.csv")
    print(df_yeast.head())
    print("Shape of df_yeast: {}\n".format(df_yeast.shape))

    # Doing some data cleaning here:
    df_yeast.rename(columns={"kill_volume": "ethanol"}, inplace=True)
    df_yeast.drop(columns=["well", "Time", "(conc, time)"], inplace=True)
    df_yeast["stain"] = df_yeast["stain"].map({"SYTOX Red Stain": 1})
    df_yeast["stain"].fillna(0, inplace=True)
    df_yeast["stain"] = df_yeast["stain"].astype(int)
    col_order = list(df_yeast.columns.values)
    col_order.insert(1, col_order.pop(col_order.index('stain')))
    df_yeast = df_yeast[col_order]

    print(df_yeast)

    # df_yeast.to_csv("full_yeast_df_cleaned.csv", index=False)


if __name__ == '__main__':
    main()
