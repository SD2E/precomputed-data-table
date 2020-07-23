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


def run_wasserstain_analysis(er_dir, r_dict):

    for datafile in r_dict.keys():
        wasser_df = wasser.do_analysis(er_dir, datafile)
        r_dict[datafile]['wasserstein_dists'] = wasser_df
    return r_dict


def return_fc_meta_name(er_dir):

    for fname in os.listdir(er_dir):
        if fname.endswith("_fc_meta.csv"):
            return os.path.realpath(os.path.join(er_dir, fname))


def get_col_min_max(er_dir, column):

    meta_filename = return_fc_meta_name(er_dir)
    meta_df = pd.read_csv(meta_filename)
    return  [meta_df[column].min(), meta_df[column].max()]


# ten-fold change within a single well
def run_tenfold_time_diff(er_dir, r_dict, meta_fname, exp_ref):

    groupby_columns = ['strain', 'inducer_concentration', 'experiment_id', 'well']
    comparison_column = 'timepoint'
    comparison_values = get_col_min_max(er_dir, comparison_column)

    for datafile, single_results_dict in r_dict.items():
        wasser_results_df = r_dict[datafile]['wasserstein_dists']
        summary, _ = tenfoldcomp.compute_difference(wasser_results_df, meta_fname, groupby_columns,
                                                    comparison_column, comparison_values)

        summary = summary.reset_index()
        summary[groupby_columns] = pd.DataFrame(summary['index'].tolist())
        summary = summary.drop('index', axis=1)
        summary['experiment_reference'] = exp_ref

        r_dict[datafile]['time_diff']['tenfold_summary'] = summary

        params = {"results_file": datafile, "metadata_file": meta_fname,
                  "groupby_columns": groupby_columns, "comparison_column": comparison_column,
                  "comparison_values": comparison_values}
        r_dict[datafile]['time_diff']['tenfold_params'] = params

    return r_dict


# ten-fold change across inducers
def run_tenfold_inducer_diff(er_dir, r_dict, meta_fname, exp_ref):

    groupby_columns = ["strain", "timepoint", "experiment_id"]
    comparison_column = "inducer_concentration"
    comparison_values = get_col_min_max(er_dir, comparison_column)

    for datafile, single_results_dict in r_dict.items():
        wasser_results_df = r_dict[datafile]['wasserstein_dists']
        summary, _ = tenfoldcomp.compute_difference(wasser_results_df, meta_fname, groupby_columns,
                                                    comparison_column, comparison_values)

        summary = summary.reset_index()
        summary[groupby_columns] = pd.DataFrame(summary['index'].tolist())
        summary = summary.drop('index', axis=1)
        summary['experiment_reference'] = exp_ref

        r_dict[datafile]['inducer_diff']['tenfold_summary'] = summary

        params = {"results_file": datafile, "metadata_file": meta_fname,
                  "groupby_columns": groupby_columns, "comparison_column": comparison_column,
                  "comparison_values": comparison_values}
        r_dict[datafile]['inducer_diff']['tenfold_params'] = params

    return r_dict


def run_wasser_tenfold(exp_ref, exp_ref_dir):

    meta_fname = return_fc_meta_name(exp_ref_dir)

    data_dict = {'wasserstein_dists': '',
                 'time_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                 'inducer_diff': {'tenfold_summary': '', 'tenfold_params': ''}}#,
                 # 'control_diff': {'tenfold_summary': '', 'tenfold_params': ''}}

    results_dict = {"fc_raw_log10_stats.csv": data_dict,
                    "fc_etl_stats.csv": data_dict}

    # results_dict = run_wasserstain_analysis(exp_ref_dir, results_dict)
    # results_dict = run_tenfold_time_diff(exp_ref_dir, results_dict, meta_fname, exp_ref)
    # results_dict = run_tenfold_inducer_diff(exp_ref_dir, results_dict, meta_fname, exp_ref)

    fname_dict = {}
    datetime = subprocess.check_output(['date +%Y_%m_%d_%H_%M_%S'], shell=True).decode(sys.stdout.encoding).strip()

    for datafile, single_results_dict in results_dict.items():
        full_datafile_name = '__'.join([exp_ref, datafile])
        fname_dict[full_datafile_name] = []
        for data in single_results_dict.keys():
            if data == 'wasserstein_dists':
                fname_wasser = 'pdt_{}__{}_{}_{}.csv'.format(exp_ref, datafile.split(".")[0], data, datetime)
                results_dict = run_wasserstain_analysis(exp_ref_dir, results_dict)
                results_dict[datafile]['wasserstein_dists'].to_csv(fname_wasser)
                fname_dict[full_datafile_name].append(fname_wasser)
            # else:
            #     fname_summary = 'pdt_{}__{}_{}_summary_{}.csv'.format(exp_ref, datafile.split(".")[0], data, datetime)
            #     fname_params = 'pdt_{}__{}_{}_params_{}.json'.format(exp_ref, datafile.split(".")[0], data, datetime)
            #     tenfoldcomp.save_summary(results_dict[datafile][data]['tenfold_summary'], fname_summary)
            #     tenfoldcomp.save_params(results_dict[datafile][data]['tenfold_params'], fname_params)
            #     fname_dict[full_datafile_name].append(fname_summary)
            #     fname_dict[full_datafile_name].append(fname_params)

    for datafile, single_results_dict in results_dict.items():
        full_datafile_name = '__'.join([exp_ref, datafile])
        fname_dict[full_datafile_name] = []
        for data in single_results_dict.keys():
            if data == 'time_diff':
                fname_summary = 'pdt_{}__{}_{}_summary_{}.csv'.format(exp_ref, datafile.split(".")[0], data, datetime)
                fname_params = 'pdt_{}__{}_{}_params_{}.json'.format(exp_ref, datafile.split(".")[0], data, datetime)
                results_dict = run_tenfold_time_diff(exp_ref_dir, results_dict, meta_fname, exp_ref)
                tenfoldcomp.save_summary(results_dict[datafile][data]['tenfold_summary'], fname_summary)
                tenfoldcomp.save_params(results_dict[datafile][data]['tenfold_params'], fname_params)
                fname_dict[full_datafile_name].append(fname_summary)
                fname_dict[full_datafile_name].append(fname_params)
            elif data == 'inducer_diff':
                fname_summary = 'pdt_{}__{}_{}_summary_{}.csv'.format(exp_ref, datafile.split(".")[0], data, datetime)
                fname_params = 'pdt_{}__{}_{}_params_{}.json'.format(exp_ref, datafile.split(".")[0], data, datetime)
                results_dict = run_tenfold_inducer_diff(exp_ref_dir, results_dict, meta_fname, exp_ref)
                tenfoldcomp.save_summary(results_dict[datafile][data]['tenfold_summary'], fname_summary)
                tenfoldcomp.save_params(results_dict[datafile][data]['tenfold_params'], fname_params)
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
