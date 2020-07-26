"""
run run_wasserstein_tenfold_comparisons.py on flow cytometry data frames produced by Data Converge;

example command:
python run_wasserstein_tenfold_comparisons.py "YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208"

:authors: Bree Cummins & Robert C. Moseley (robert.moseley@duke.edu)
"""

import os
import argparse
import subprocess
import sys
import pandas as pd
import wasserstein_tenfold_comparisons.wasserstein_analysis as wasser
import wasserstein_tenfold_comparisons.TenFoldComparisons as tenfoldcomp


def run_wasserstain_analysis(er_dir, datafile):

    wasser_df = wasser.do_analysis(er_dir, datafile)
    return wasser_df


def return_fc_meta_name(er_dir):

    for fname in os.listdir(er_dir):
        if fname.endswith("_fc_meta.csv"):
            return os.path.realpath(os.path.join(er_dir, fname))


def get_col_min_max(er_dir, column):

    meta_filename = return_fc_meta_name(er_dir)
    meta_df = pd.read_csv(meta_filename)
    return [meta_df[column].min(), meta_df[column].max()]


diff_dict = {'inducer_diff':
                 {'comparison_col': 'inducer_concentration',
                  'groupby': ["strain", "timepoint", "experiment_id"]},
             'time_diff':
                 {'comparison_col': 'timepoint',
                  'groupby': ['strain', 'inducer_concentration', 'experiment_id', 'well']},
             'time_reps_diff':
                 {'comparison_col': 'timepoint',
                  'groupby': ['strain', 'inducer_concentration']}}


def tenfold_comparison(er_dir, r_dict, datafile, diff_name, meta_fname, exp_ref):

    groupby_columns = diff_dict[diff_name]['groupby']
    comparison_column = diff_dict[diff_name]['comparison_col']
    comparison_values = get_col_min_max(er_dir, comparison_column)

    wasser_results_df = r_dict[datafile]['wasserstein_dists']
    summary, _ = tenfoldcomp.compute_difference(wasser_results_df, meta_fname, groupby_columns,
                                                comparison_column, comparison_values)

    summary = summary.reset_index()
    summary[groupby_columns] = pd.DataFrame(summary['index'].tolist())
    summary = summary.drop('index', axis=1)
    summary['experiment_reference'] = exp_ref
    groupby_col = pd.Series([groupby_columns] * len(summary))
    summary['group_name'] = groupby_col

    r_dict[datafile][diff_name]['tenfold_summary'] = summary

    params = {"results_file": datafile, "metadata_file": meta_fname,
              "groupby_columns": groupby_columns, "comparison_column": comparison_column,
              "comparison_values": comparison_values}
    r_dict[datafile][diff_name]['tenfold_params'] = params

    return r_dict


def run_wasser_tenfold(exp_ref, exp_ref_dir):

    meta_fname = return_fc_meta_name(exp_ref_dir)

    results_dict = {"fc_raw_log10_stats.csv": {'wasserstein_dists': '',
                                               'time_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                               'inducer_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                               'time_reps_diff': {'tenfold_summary': '', 'tenfold_params': ''}},
                    "fc_etl_stats.csv": {'wasserstein_dists': '',
                                         'time_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                         'inducer_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                         'time_reps_diff': {'tenfold_summary': '', 'tenfold_params': ''}}}


    fname_dict = {}
    datetime = subprocess.check_output(['date +%Y_%m_%d_%H_%M_%S'], shell=True).decode(sys.stdout.encoding).strip()

    for datafile, single_results_dict in results_dict.items():
        full_datafile_name = '__'.join([exp_ref, datafile])
        fname_dict[full_datafile_name] = []
        for data in single_results_dict.keys():
            if data == 'wasserstein_dists':
                fname_wasser = 'pdt_{}__{}_{}_{}.csv'.format(exp_ref, datafile.split(".")[0], data, datetime)
                results_dict[datafile]['wasserstein_dists'] = run_wasserstain_analysis(exp_ref_dir, datafile)
                results_dict[datafile]['wasserstein_dists'].to_csv(fname_wasser)
                fname_dict[full_datafile_name].append(fname_wasser)

    for datafile, single_results_dict in results_dict.items():
        full_datafile_name = '__'.join([exp_ref, datafile])
        fname_dict[full_datafile_name] = []
        for diff_name in diff_dict.keys():

            fname_summary = 'pdt_{}__{}_{}_summary_{}.csv'.format(exp_ref, datafile.split(".")[0], diff_name, datetime)
            fname_params = 'pdt_{}__{}_{}_params_{}.json'.format(exp_ref, datafile.split(".")[0], diff_name, datetime)
            results_dict = tenfold_comparison(exp_ref_dir, results_dict, datafile, diff_name, meta_fname, exp_ref)
            tenfoldcomp.save_summary(results_dict[datafile][diff_name]['tenfold_summary'], fname_summary)
            tenfoldcomp.save_params(results_dict[datafile][diff_name]['tenfold_params'], fname_params)
            fname_dict[full_datafile_name].append(fname_summary)
            fname_dict[full_datafile_name].append(fname_params)

    return fname_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_ref', help='experimental reference from data science table')
    parser.add_argument('--exp_ref_dir', help='path to experimental reference directory')

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_exp_ref_dir = args.exp_ref_dir

    run_wasser_tenfold(arg_exp_ref, arg_exp_ref_dir)
