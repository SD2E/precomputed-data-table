"""
run run_wasserstein_tenfold_comparisons.py on flow cytometry data frames produced by Data Converge;

example command:
python run_wasserstein_tenfold_comparisons.py "YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208"

:authors: Bree Cummins & Robert C. Moseley (robert.moseley@duke.edu)
"""

import argparse
import os
import json
import subprocess
import sys
import pandas as pd
import wasserstein_analysis as wasser
import TenFoldComparisons as tenfoldcomp
from common import record_product_info as rpi
from common import preproc

def run_wasserstain_analysis(er_dir, datafile):

    wasser_df = wasser.do_analysis(er_dir, datafile)
    return wasser_df


def return_fc_meta_name(er_dir):

    for fname in os.listdir(er_dir):
        if fname.endswith("_fc_meta.csv"):
            return os.path.realpath(os.path.join(er_dir, fname))


def get_col_min_max(er_dir, column):

    meta_filename = return_fc_meta_name(er_dir)
    meta_df = pd.read_csv(meta_filename, dtype=object)
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


def tenfold_lookup(er_dir, r_dict, datafile, diff_name, meta_fname, exp_ref):
    '''
    Inputs:
    er_dir: directory of  experimental reference data in Data Converge
    r_dict: dictionary for connecting analyses with their results
    datafile: the data type being analyzed, either fc_raw_log10_stats or fc_etl_stats
    diff_name: the name of the question being asked. This is used to gather the arguments used in tenfoldcomp.compute_difference from the diff_dict.
    meta_fname: name of the metadata file. The groupby is performed on this file.
    exp_ref: name of the experimental reference

    Output:
    r_dict: dictionary for connecting analyses with their results.
    '''

    groupby_columns = diff_dict[diff_name]['groupby']
    comparison_column = diff_dict[diff_name]['comparison_col']
    comparison_values = get_col_min_max(er_dir, comparison_column)

    wasser_results_df = r_dict[datafile]['wasserstein_dists']
    summary, _ = tenfoldcomp.wasserstein_lookup(wasser_results_df, meta_fname, groupby_columns,
                                                comparison_column, comparison_values)

    summary = summary.reset_index()
    summary[groupby_columns] = pd.DataFrame(summary['index'].tolist())
    summary = summary.rename(columns={'index': 'group_name'})
    summary['group_name'] = summary['group_name'].apply(lambda x: tuple([str(s) for s in x]))
    summary['experiment_reference'] = exp_ref

    r_dict[datafile][diff_name]['tenfold_summary'] = summary

    params = {"results_file": datafile, "metadata_file": meta_fname,
              "groupby_columns": groupby_columns, "comparison_column": comparison_column,
              "comparison_values": comparison_values}
    r_dict[datafile][diff_name]['tenfold_params'] = params

    return r_dict


def flow_file_check(er_dir):
    raw_log10_file = False
    etl_stats_file = False

    for fname in os.listdir(er_dir):
        if fname.endswith("_fc_raw_log10_stats.csv"):
            raw_log10_file = True
        if fname.endswith("_fc_etl_stats.csv"):
            etl_stats_file = True

    return raw_log10_file, etl_stats_file


def run_wasser_tenfold(exp_ref, exp_ref_dir):

    meta_fname = return_fc_meta_name(exp_ref_dir)

    raw_log10_file_exist, etl_stats_file_exist = flow_file_check(exp_ref_dir)

    results_dict = {}

    if raw_log10_file_exist:
        results_dict["fc_raw_log10_stats.csv"] = {'wasserstein_dists': '',
                                                  'time_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                                  'inducer_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                                  'time_reps_diff': {'tenfold_summary': '', 'tenfold_params': ''}}
    if etl_stats_file_exist:
        results_dict["fc_etl_stats.csv"] = {'wasserstein_dists': '',
                                            'time_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                            'inducer_diff': {'tenfold_summary': '', 'tenfold_params': ''},
                                            'time_reps_diff': {'tenfold_summary': '', 'tenfold_params': ''}}

    fname_dict = {}
    datetime = subprocess.check_output(['date +%Y_%m_%d_%H_%M_%S'], shell=True).decode(sys.stdout.encoding).strip()

    for datafile, single_results_dict in results_dict.items():
        full_datafile_name = '__'.join([exp_ref, datafile])
        if full_datafile_name not in fname_dict.keys():
            fname_dict[full_datafile_name] = []

        for data in single_results_dict.keys():
            if data == 'wasserstein_dists':
                fname_wasser = 'pdt_{}__{}_{}_{}.csv'.format(exp_ref, datafile.split(".")[0], data, datetime)
                results_dict[datafile]['wasserstein_dists'] = run_wasserstain_analysis(exp_ref_dir, datafile)
                results_dict[datafile]['wasserstein_dists'].to_csv(fname_wasser)
                fname_dict[full_datafile_name].append(fname_wasser)

    for datafile, single_results_dict in results_dict.items():
        full_datafile_name = '__'.join([exp_ref, datafile])
        if full_datafile_name not in fname_dict.keys():
            fname_dict[full_datafile_name] = []

        for diff_name in diff_dict.keys():
            fname_summary = 'pdt_{}__{}_{}_summary_{}.csv'.format(exp_ref, datafile.split(".")[0], diff_name, datetime)
            fname_params = 'pdt_{}__{}_{}_params_{}.json'.format(exp_ref, datafile.split(".")[0], diff_name, datetime)
            results_dict = tenfold_lookup(exp_ref_dir, results_dict, datafile, diff_name, meta_fname, exp_ref)
            tenfoldcomp.save_summary(results_dict[datafile][diff_name]['tenfold_summary'], fname_summary)
            tenfoldcomp.save_params(results_dict[datafile][diff_name]['tenfold_params'], fname_params)
            fname_dict[full_datafile_name].append(fname_summary)
            fname_dict[full_datafile_name].append(fname_params)

    return fname_dict

def main(exp_ref, analysis, out_dir, data_converge_dir):

    # Check status of data in ER's record.json file
    path_to_record_json = preproc.return_er_record_path(data_converge_dir)
    preproc.check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    data_confirm_dict = preproc.confirm_data_types(os.listdir(data_converge_dir))

    record_path = os.path.join(out_dir, "record.json")

    record = {}

    wasser_fname_dict = run_wasser_tenfold(exp_ref, data_converge_dir)
    record = rpi.append_record(record, wasser_fname_dict, analysis, out_dir)

    with open(record_path, 'w') as jfile:
        json.dump(record, jfile, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_ref", help="experimental reference from data science table")
    parser.add_argument("--data_converge_dir", help="path to Data Converge directory")
    parser.add_argument("--analysis", help="analysis to run")

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_data_converge_dir = args.data_converge_dir
    arg_analysis = args.analysis
    arg_out_dir = "."

    main(arg_exp_ref, arg_analysis, arg_out_dir, arg_data_converge_dir)
