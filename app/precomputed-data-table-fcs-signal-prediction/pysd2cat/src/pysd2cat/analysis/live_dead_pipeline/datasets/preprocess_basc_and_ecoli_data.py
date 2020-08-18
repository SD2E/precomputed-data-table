import os
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    df = pd.read_csv("LD_Microbe.csv")
    df.drop(columns=["Unnamed: 0", "ethanol_unit", "sample_id", "replicate"], inplace=True)
    df.rename(columns={"ethanol_concentration": "ethanol"}, inplace=True)
    df["time_point"] = ((df["time_point"] - 22) * 2).astype(int)

    # getting rid of the 35.0 ethanol concentration because there aren't many samples/events
    # for that concentration, both in bacillus and ecoli.
    df = df.loc[df["ethanol"] != 35.0]

    df_basc = df.loc[df["strain"] == "Bacillus subtilis 168 Marburg"]
    df_ecoli = df.loc[df["strain"] == "MG1655_WT"]

    col_order = ['arbitrary_index', 'stain', 'ethanol', 'time_point',
                 'FSC-A', 'SSC-A', 'RL1-A', 'FSC-H', 'SSC-H', 'RL1-H', 'FSC-W', 'SSC-W', 'RL1-W']
    print(df_basc.shape)
    print(df_ecoli.shape)
    print(len(col_order))
    print()
    df_basc = df_basc[col_order]
    df_ecoli = df_ecoli[col_order]
    print(df_basc.shape)
    print(df_ecoli.shape)
    print()
    print(df_basc)
    print()
    print(df_ecoli)

    # df_basc.to_csv("full_basc_df_cleaned.csv", index=False)
    # df_ecoli.to_csv("full_ecoli_df_cleaned.csv", index=False)


if __name__ == '__main__':
    main()
