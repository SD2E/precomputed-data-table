#!/usr/bin/env python
"""
:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import os
import argparse
from os.path import expanduser
from grouped_control_prediction.main import main
import pandas as pd
import numpy as np


def run_live_dead_prediction(exp_ref, exp_dir):


    # Set model parameters
    project_id = "sd2e-project-14"
    low_control = "CRISPR_CEN.PK2_negative_control_23970"
    high_control = "CRISPR_CEN.PK2_positive_control_NOR_00_24864"
    weighted_controls = True
    wass_path = "data/" + prediction_dataset + ".pkl"
    control_size = 10

    result, rf_df, test_accuracy, timeseries_fig, samples_and_controls_fig = main(exp_dir,
                                                                                  project_id,
                                                                                  low_control,
                                                                                  high_control,
                                                                                  weighted_controls,
                                                                                  wass_path,
                                                                                  control_size)

    # results_fname = 'pdt_{}__fcs_signal_prediction.csv'.format(exp_ref)
    #
    # record = du.get_record(exp_dir)
    # dc_raw_events_dict = du.get_record_file(record, file_type="fc_raw_events")
    # dc_input_fname = dc_raw_events_dict['name']
    #
    # return result, results_fname, dc_input_fname


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_ref', help='experimental reference from data science table')
    parser.add_argument('--exp_ref_dir', help='path to experimental reference directory')

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_exp_ref_dir = args.exp_ref_dir

    run_live_dead_prediction(arg_exp_ref, arg_exp_ref_dir)
