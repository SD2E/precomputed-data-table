#!/usr/bin/env python
"""
:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

from fcs_signal_prediction.src.fcs_signal_prediction.main import main
from fcs_signal_prediction.src.fcs_signal_prediction.utils import data_utils as du
import argparse


def run_fcs_signal_prediction(exp_ref, exp_dir, out_dir):

    high_control = 'CRISPR_CEN.PK2_positive_control_NOR_00_24864'
    low_control = 'CRISPR_CEN.PK2_negative_control_23970'

    result, timeseries_fig, samples_and_controls_fig = main(exp_dir,
                                                            exp_ref,
                                                            low_control,
                                                            high_control)

    results_fname = 'pdt_{}__fcs_signal_prediction.csv'.format(exp_ref)

    record = du.get_record(exp_dir)
    dc_input_fname = du.get_record_file(record, file_type="fc_raw_events")

    return result, results_fname, dc_input_fname


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_ref', help='experimental reference from data science table')
    parser.add_argument('exp_ref_dir', help='path to experimental reference directory')
    # parser.add_argument("output_dir", help="directory where to write the output files")

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_exp_ref_dir = args.exp_ref_dir
    # arg_out_dir = args.output_dir

    run_fcs_signal_prediction(arg_exp_ref, arg_exp_ref_dir)