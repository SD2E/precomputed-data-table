#!/usr/bin/env python
"""
:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import argparse
import os
import json
from itertools import product
from fcs_signal_prediction.main import main as fsp
from fcs_signal_prediction.utils import data_utils as du
from common import record_product_info as rpi
from common import preproc


def get_controls(exp_dir):

    meta_df = du.get_meta(exp_dir, du.get_record(exp_dir))

    high_controls_df = meta_df[meta_df['control_type'] == 'HIGH_FITC']
    high_controls = list(set(high_controls_df['strain_name'].values))

    low_controls_df = meta_df[meta_df['control_type'] == 'EMPTY_VECTOR']
    low_controls = list(set(low_controls_df['strain_name'].values))

    return high_controls, low_controls

def run_fcs_signal_prediction(exp_ref, exp_dir):

    # get unique combos of high/low controls and analyze all combos
    # high_control = 'CRISPR_CEN.PK2_positive_control_NOR_00_24864'
    # low_control = 'CRISPR_CEN.PK2_negative_control_23970'
    high_controls_list, low_controls_list = get_controls(exp_dir)
    print("high: {}".format(high_controls_list))
    print("low: {}".format(low_controls_list))

    results_dict = {}
    hl_combinations = list(product(high_controls_list, low_controls_list))

    for idx, combo in enumerate(hl_combinations):
        high_control = combo[0]
        low_control = combo[1]

        result = fsp(exp_dir, exp_ref, low_control, high_control)
        result['high_control'] = high_control
        result['low_control'] = low_control
        # timeseries_fig.savefig('{}__well_timeseries_figure.png'.format(exp_ref), format='png', dpi=100)
        # samples_and_controls_fig.savefig('{}__samples_and_controls_figure.png'.format(exp_ref), format='png', dpi=100)

        results_fname = 'pdt_{}__HL{}_fcs_signal_prediction.csv'.format(exp_ref, idx+1)
        results_dict[results_fname] = result

    record = du.get_record(exp_dir)
    dc_raw_events_dict = du.get_record_file(record, file_type="fc_raw_events")
    dc_input_fname = dc_raw_events_dict['name']

    return results_dict, dc_input_fname


def main(exp_ref, analysis, out_dir, data_converge_dir):

    # Check status of data in ER's record.json file
    path_to_record_json = preproc.return_er_record_path(data_converge_dir)
    preproc.check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    data_confirm_dict = preproc.confirm_data_types(os.listdir(data_converge_dir))

    record_path = os.path.join(out_dir, "record.json")

    fcs_result_dict, raw_event_fname = run_fcs_signal_prediction(exp_ref, data_converge_dir)
    for results_name, results_df in fcs_result_dict.items():
        results_df.to_csv(results_name, index=False)

    fcs_signal_dict = {raw_event_fname: list(fcs_result_dict.keys())}
    record = {}
    record = rpi.append_record(record, fcs_signal_dict, analysis, out_dir)

    with open(record_path, 'w') as jfile:
        json.dump(record, jfile, indent=2)

if __name__ == "__main__":
    
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

