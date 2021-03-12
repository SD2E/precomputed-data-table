#!/usr/bin/env python
"""
:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from os.path import expanduser
from grouped_control_prediction.main import main as gcpm
from grouped_control_prediction.utils import data_utils as du
from common import record_product_info as rpi
from common import preproc

def run_live_dead_prediction(exp_ref, exp_dir, control_set_dir):

    # add control_set_dir as input to main
    
    # Set model parameters
    project_id = "sd2e-project-14"
    low_control = "CRISPR_CEN.PK2_negative_control_23970"
    high_control = "CRISPR_CEN.PK2_positive_control_NOR_00_24864"
    weighted_controls = True
    wass_path = "blank"
    control_size = 10

    result, rf_df, test_accuracy = gcpm(exp_dir,
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


def main(exp_ref, analysis, out_dir, data_converge_dir, control_set_dir):

    # Check status of data in ER's record.json file
    path_to_record_json = preproc.return_er_record_path(data_converge_dir)
    preproc.check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    #data_confirm_dict = preproc.confirm_data_types(os.listdir(data_converge_dir))

    record_path = os.path.join(out_dir, "record.json")

    record = {}

    ld_pred_result_df, ld_pred_results_fname, raw_event_fname = run_live_dead_prediction(exp_ref, data_converge_dir, control_set_dir)
    ld_pred_result_df.to_csv(ld_pred_results_fname, index=False)
    ld_pred_signal_dict = {raw_event_fname: [ld_pred_results_fname]}
    record = rpi.append_record(record, ld_pred_signal_dict, analysis, out_dir)

    with open(record_path, 'w') as jfile:
        json.dump(record, jfile, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_ref", help="experimental reference from data science table")
    parser.add_argument("--data_converge_dir", help="path to Data Converge directory")
    parser.add_argument("--control_set_dir", help="path to data for creating control sets")
    parser.add_argument("--analysis", help="analysis to run")

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_data_converge_dir = args.data_converge_dir
    arg_control_set_dir = args.control_set_dir
    arg_analysis = args.analysis
    arg_out_dir = "."

    main(arg_exp_ref, arg_analysis, arg_out_dir, arg_data_converge_dir, arg_control_set_dir)
