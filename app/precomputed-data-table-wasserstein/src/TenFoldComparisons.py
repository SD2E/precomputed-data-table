import pandas as pd
import json
from numpy import nan
from statistics import median


def load_csv(fname):
    # for both results and combined metadata df
    return pd.read_csv(fname)

def group_me(df,columns):
    # for metadata df
    return df.groupby(columns)

def get_samp_id_order(df):
    # for results df
    return list(df.columns)[1:]

def get_matrix(df):
    #for results df
    return df.values

def get_samp_ids(df,col,value):
    # for group in grouped metadata df
    df2 = df[df[col] == value]
    return df2["sample_id"].values

def get_index_pair_value(results,samp_id_order,id1,id2):
    # for results matrix from results df
    ind1 = samp_id_order.index(id1)
    ind2 = samp_id_order.index(id2)
    return results[ind1,ind2]


def wasserstein_lookup(results_df, metadata_file, columns, comparison_column, comparison_values):
    '''
    results_file = Wasserstein results
    metadata_file = csv with merged metadata columns
    columns = columns to groupby (probably strain, BE conc, and rep)
    comparison_column = column containing the two values to compare
    comparison_values = 2 vector with a start value and an end value
    threshold_10_fold = number above which a minimum of a 10-fold difference is detected (i.e. may want to choose 9.5 instead of 10)
    '''
    meta_df = pd.read_csv(metadata_file, dtype=object)
    meta_grouped = group_me(meta_df, columns)
    # res_df = load_csv(results_file)
    samp_id_order = get_samp_id_order(results_df)
    results = get_matrix(results_df)[:, 1:]
    values = {}
    missing = []
    for name, group in meta_grouped:
        initial_ids = get_samp_ids(group, comparison_column, comparison_values[0])
        final_ids = get_samp_ids(group, comparison_column, comparison_values[1])
        group_vals = []
        # pairs = []
        samp_id_init_list = []
        samp_id_fin_list = []
        for samp_id_init in initial_ids:
            for samp_id_fin in final_ids:
                try:
                    val = get_index_pair_value(results, samp_id_order, samp_id_init, samp_id_fin)
                except:
                    # handle dropped samples in ETL
                    # print("One of the two sample ids {} not found.".format([samp_id_init,samp_id_fin]))
                    val = nan
                group_vals.append(val)
                # pairs.append([samp_id_init, samp_id_fin])
                samp_id_init_list.append(samp_id_init)
                samp_id_fin_list.append(samp_id_fin)
        if group_vals:
            # values[name] = [min(group_vals), median(group_vals), max(group_vals), pairs]
            values[name] = [min(group_vals), median(group_vals), max(group_vals), samp_id_init_list, samp_id_fin_list]
        else:
            missing.append(name)
    summary = pd.DataFrame.from_dict(values, orient='index',
                                     columns=['wasserstein_min', 'wasserstein_median', 'wasserstein_max',
                                              'sample_id-init', 'sample_id-fin'])

    summary['sample_id-init'] = summary['sample_id-init'].apply(lambda x: x[0])
    summary['sample_id-fin'] = summary['sample_id-fin'].apply(lambda x: x[0])

    return summary, missing


def save_summary(summary, fname):
    summary.to_csv(fname)


def save_params(params, fname):
    json.dump(params, open(fname, "w"))
