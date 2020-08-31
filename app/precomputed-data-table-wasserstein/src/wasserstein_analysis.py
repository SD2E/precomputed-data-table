import pyemd
import pandas as pd
import os
import json
import numpy as np


def emdist(h1, h2, bin_vals):
    '''
    Calculate earth mover's distance between 2 histograms. Histograms are normalized to mass 1.
    :param h1: a 1-D numpy array representing a histogram with the bin values used to make bin_dist
    :param h2: a 1-D numpy array representing a histogram with the bin values used to make bin_dist
    :param bin_vals: representative points in the bins defining the histograms in data
    :return: a scalar value that is the earth mover's distance between normalized h1 and h2
    '''
    bin_dist = np.array([[np.abs(m - n) for m in bin_vals] for n in bin_vals])
    if float(sum(h1)) > 0 and float(sum(h2)) > 0 and not any(np.isnan(h1)) and not any(np.isnan(h1)):
        return pyemd.emd(np.asarray(h1)/float(sum(h1)), np.asarray(h2)/float(sum(h2)), bin_dist)
    else:
        return np.nan


def get_record(experiment):
    record = json.load(open(os.path.join(experiment, "record.json")))
    return record


def get_record_file(record, file_type="fc_meta"):
    files = record['files']
    files_of_type = [ x for x in files if file_type in x['name']]
    if len(files_of_type) > 0:
        return files_of_type[0]
    else:
        return None


def get_data(experiment, record, file_type):
    fc_raw_file = get_record_file(record, file_type)
    if fc_raw_file:
        data_df = pd.read_csv(os.path.join(experiment, fc_raw_file['name']))
        return data_df
    else:
        return None


def get_bins(df):
    return [float(x.split("_")[1]) for x in df.columns if "bin" in x]


def get_row_values(df,row_name,id_col):
    df_j = df.loc[df[id_col] == row_name]
    df_j = df_j[[x for x in df_j.columns if "bin" in x]]
    return df_j.values[0]


def do_analysis(experiment, datafile, id_col="sample_id", channel_col="channel", channel_val="BL1-A"):
    # datafile is "fc_raw_log10_stats.csv" or "fc_etl_stats.csv"

    ## load dataset from data converge
    record = get_record(experiment)
    df = get_data(experiment, record, datafile)

    if df is None:
        return None

    # handle difference between etl and log10 histogram filesS
    if channel_col in df.columns:
        df = df.loc[df[channel_col] == channel_val]

    bins = get_bins(df)
    ids = list(df[id_col].values)
    res = np.zeros([len(ids), len(ids)])

    for j, s in enumerate(ids):
        s_bin_vals = get_row_values(df, s, id_col)
        for k, t in enumerate(ids[j + 1:]):
            t_bin_vals = get_row_values(df, t, id_col)
            score = emdist(s_bin_vals, t_bin_vals, bins)
            res[j, j + k + 1] = 10 ** score
            res[j + k + 1, j] = 10 ** score
    df_results = pd.DataFrame(data=res, index=ids, columns=ids)
    return df_results