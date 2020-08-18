"""
metrics of ON vs OFF for plate reader or aggregated flow cytometry data (one measurement per sample).

:authors: anastasia deckard anastasia<dot>deckard<at>geomdata<dot>com
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib
import platform
if platform.system() == "Darwin":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# column mappings, in case data format changes
experiment_col = 'experiment_id'
strain_col = 'strain'
time_series_col = 'output_id'
replicate_col = 'replicate'
observed_output_col = 'observed_fluor'
intended_output_col = 'intended_output'


def make_synth_data():
    """
    makes a synthetic data set in a narrow format. synthetic data is plate reader data.

    :return: pandas.DataFrame, columns: ['experiment_id', 'output_id', 'time', 'replicate', 'intended_output',
       'input', 'observed_fluor']

    """
    # make up some time series input/output patterns
    times = [1, 2, 3]
    ts_io = {
        'ts1': {'i': ['01', '00', '00'],
                'o': ['0', '1', '1']},
        'ts2': {'i': ['00', '01', '01'],
                'o': ['1', '0', '0']}}

    details = {
        'experiment_id': ['exp1', 'exp2'],  # this is a group of time series?
        'strain': ['UWBF1', 'UWBF2'],  # this is the circuit
        'output_id': ['ts1', 'ts2'],  # this is one time series?
        'time': times,
        'replicate': [1, 2],
    }

    keys, values = zip(*details.items())
    records = [dict(zip(keys, x)) for x in itertools.product(*values, repeat=1)]

    # make some clean 0 = low and 1 = high records, low noise.
    for record in records:
        record['input'] = ts_io[record['output_id']]['i'][times.index(record['time'])]
        record['intended_output'] = ts_io[record['output_id']]['o'][times.index(record['time'])]
        if record['intended_output'] == '0':
            mu, sigma = 100, 25
        else:
            mu, sigma = 1000, 250
        observed_fluor = np.random.normal(mu, sigma, 1)[0]
        record.update({'observed_fluor': observed_fluor})

    data_df = pd.DataFrame(records)

    return data_df


def min_max_diff(data_df, group_cols):
    """
    measure fold and absolute change between
    the minimum GFP output for ON states vs the maximum GFP output for OFF states,
    for specified grouping of columns (exp, ts, etc)

    :param data_df: pandas.DataFrame
    :param group_cols: list of columns to group by
    :return: pandas.DataFrame
    """

    # group by the exp and ts ids
    records = list()
    grouped_df = data_df.groupby(group_cols)
    for name, group in grouped_df:
        # compute max of OFF and min of ON
        off_max = group[(group[intended_output_col] == '0')][observed_output_col].max()
        on_min = group[(group[intended_output_col] == '1')][observed_output_col].min()
        # compute absolute difference
        diff_min_on_max_off_abs = on_min - off_max
        # compute fold change
        diff_min_on_max_off_fc = on_min / off_max  # need to catch zero here

        # make record
        record = dict(zip(group_cols, name))
        record_results = {'off_max': off_max,
                          'on_min': on_min,
                          'd_mn1_mx0_abs': diff_min_on_max_off_abs,
                          'd_mn1_mx0_fc': diff_min_on_max_off_fc}
        record.update(record_results)
        records.append(record)

    records_df = pd.DataFrame(records)
    records_df.sort_values(by=group_cols,
                           inplace=True)

    return records_df


def min_max_diff_strain(data_df):
    """
    measure fold and absolute change between
    the minimum GFP output for ON states vs the maximum GFP output for OFF states,
    for each experiment, strain; combine all time series, replicates, and time points
    so group by experiment, strain

    :param data_df: pandas.DataFrame
    :return: pandas.DataFrame
    """

    group_cols = [experiment_col, strain_col]
    results_df = min_max_diff(data_df=data_df, group_cols=group_cols)

    return results_df


def min_max_diff_ts(data_df):
    """
    measure fold and absolute change between
    the minimum GFP output for ON states vs the maximum GFP output for OFF states,
    for each experiment, strain, time series; combine all replicates and time points
    so group by experiment, strain, time series

    :param data_df: pandas.DataFrame
    :return: pandas.DataFrame
    """

    group_cols = [experiment_col, strain_col, time_series_col]
    results_df = min_max_diff(data_df=data_df, group_cols=group_cols)

    return results_df


def min_max_diff_replicate(data_df):
    """
    measure fold and absolute change between
    the minimum GFP output for ON states vs the maximum GFP output for OFF states,
    for each experiment, strain, time series; combine all replicates and time points
    so group by experiment, strain, time series

    :param data_df: pandas.DataFrame
    :return: pandas.DataFrame
    """

    group_cols = [experiment_col, strain_col, time_series_col, replicate_col]
    results_df = min_max_diff(data_df=data_df, group_cols=group_cols)

    return results_df


def plot_time_series(data_df):
     g = sns.FacetGrid(data_df, col=experiment_col, row=strain_col, hue=time_series_col,
                       margin_titles=True,
                       sharex=True, sharey=True,
                       legend_out=True, height=3)
     #g = g.map(plt.plot, "time", observed_output_col, marker=".")
     g.map_dataframe(sns.lineplot, x="time",
                     y=observed_output_col)

     return g.fig


def plot_scatter(data_df):
    g = sns.FacetGrid(data_df, col=experiment_col, row=strain_col, hue=time_series_col,
                      margin_titles=True,
                      sharex=True, sharey=True,
                      legend_out=True, height=3)
    # g = g.map(plt.plot, "time", observed_output_col, marker=".")
    g.map_dataframe(sns.lineplot, x="time",
                    y=observed_output_col)

    return g.fig


if __name__ == '__main__':
    data_synth_df = make_synth_data()
    results_strain_df = min_max_diff_strain(data_df=data_synth_df)
    results_timeseries_df = min_max_diff_ts(data_df=data_synth_df)
    results_replicate_df = min_max_diff_replicate(data_df=data_synth_df)

    plot = plot_time_series(data_synth_df)
    plot.savefig('../../../metrics/output/metrics_fig.png')

    print("finished")
