#!/usr/bin/env python
"""
:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import os
import argparse
import numpy as np
import pandas as pd
from os.path import expanduser
from grouped_control_prediction.main import main
from grouped_control_prediction.utils import data_utils as du


def run_live_dead_prediction(exp_ref, exp_dir, control_set_dir):

    # add control_set_dir as input to main
    
    # Set model parameters
    project_id = "sd2e-project-14"
    # low_control = "CRISPR_CEN.PK2_negative_control_23970"
    # high_control = "CRISPR_CEN.PK2_positive_control_NOR_00_24864"
    low_control = "blank"
    high_control = "blank"
    weighted_controls = True
    wass_path = "blank"
    control_size = 10

    result, rf_df, test_accuracy = main(exp_dir,
                                        project_id,
                                        low_control,
                                        high_control,
                                        weighted_controls,
                                        wass_path,
                                        control_size,
                                        control_set_dir=control_set_dir,
                                        plot=False)

    print('Test accuracy: {}'.format(test_accuracy))

    # Rename random forest dataframe columns
    rf_df.columns = list(map('_'.join, rf_df.columns.values))
    rf_df = rf_df.rename(columns={'predicted_output_mean_mean': 'RF_prediction_mean',
                                  'predicted_output_mean_std': 'RF_prediction_std'})

    results_fname = 'pdt_{}__live_dead_prediction.csv'.format(exp_ref)

    record = du.get_record(exp_dir)
    dc_raw_events_dict = du.get_record_file(record, file_type="fc_raw_events")
    dc_input_fname = dc_raw_events_dict['name']

    return rf_df, results_fname, dc_input_fname


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_ref', help='experimental reference from data science table')
    parser.add_argument('--exp_ref_dir', help='path to experimental reference directory')
    parser.add_argument("--control_set_dir", help="path to data for creating control sets")


    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_exp_ref_dir = args.exp_ref_dir
    arg_control_set_dir = args.control_set_dir

    run_live_dead_prediction(arg_exp_ref, arg_exp_ref_dir, arg_control_set_dir)
