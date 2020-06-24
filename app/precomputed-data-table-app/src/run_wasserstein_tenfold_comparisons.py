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
sys.path.insert(1, '../wasserstein_tenfold_comparisons')

from wasserstein_tenfold_comparisons import wasserstein_analysis as wasser
from wasserstein_tenfold_comparisons import TenFoldComparisons as tenfoldcomp


def run_wasserstain_analysis(er_dir, r_dict):

    for datafile in r_dict.keys():
        r_dict[datafile]['wasserstein_dists'] = wasser.do_analysis(er_dir, datafile)
    return r_dict


def return_fc_meta_name(er_dir):

    for fname in os.listdir(er_dir):
        if fname.endswith("_fc_meta.csv"):
            return os.path.realpath(os.path.join(er_dir, fname))

# ten-fold change within a single well
def run_tenfold_comparisons(er_dir, r_dict):

    meta_fname = return_fc_meta_name(er_dir)

    groupby_columns = ["strain", "inducer_concentration", 'experiment_id', 'well']
    comparison_column = "timepoint"
    comparison_values = [18.0, 24.0]
    # identifier = "_time_diff_"
    # datetime = subprocess.check_output(['date +%Y_%m_%d_%H_%M_%S'], shell=True).decode(sys.stdout.encoding).strip()

    for datafile, single_results_dict in r_dict.items():
        wasser_results_df = r_dict[datafile]['wasserstein_dists']
        summary, _ = tenfoldcomp.compute_time_difference(wasser_results_df, meta_fname, groupby_columns,
                                                         comparison_column, comparison_values)
        r_dict[datafile]['tenfold_summary'] = summary

        params = {"results_file": datafile, "metadata_file": meta_fname,
                  "groupby_columns": groupby_columns, "comparison_column": comparison_column,
                  "comparison_values": comparison_values}
        r_dict[datafile]['tenfold_params'] = params

    return r_dict

# ten-fold change across inducers
# def

def run_wasser_tenfold(exp_ref, exp_ref_dir):

    results_dict = {"fc_raw_log10_stats.csv": {'wasserstein_dists': '',
                                               'tenfold_summary': '',
                                               'tenfold_params': ''},
                    "fc_etl_stats.csv": {'wasserstein_dists': '',
                                         'tenfold_summary': '',
                                         'tenfold_params': ''}
                    }

    results_dict = run_wasserstain_analysis(exp_ref_dir, results_dict)
    results_dict = run_tenfold_comparisons(exp_ref_dir, results_dict)

    identifier = "_time_diff_"
    datetime = subprocess.check_output(['date +%Y_%m_%d_%H_%M_%S'], shell=True).decode(sys.stdout.encoding).strip()

    for datafile, single_results_dict in results_dict.items():
        fname_wasser = 'pdt_{}{}wasserstein_dists_{}.csv'.format(datafile.split(".")[0], identifier, datetime)
        fname_summary = 'pdt_{}{}summary_{}.csv'.format(datafile.split(".")[0], identifier, datetime)
        fname_params = 'pdt_{}{}params_{}.csv'.format(datafile.split(".")[0], identifier, datetime)

        results_dict[datafile]['wasserstein_dists'].to_csv(fname_wasser)
        tenfoldcomp.save_summary(results_dict[datafile]['tenfold_summary'], fname_summary)
        tenfoldcomp.save_params(results_dict[datafile]['tenfold_params'], fname_params)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_ref', help='experimental reference from data science table')
    parser.add_argument('exp_ref_dir', help='path to experimental reference directory')

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_exp_ref_dir = args.exp_ref_dir

    run_wasser_tenfold(arg_exp_ref, arg_exp_ref_dir)
