#!/usr/bin/env python
"""
:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import argparse
from itertools import product
from fcs_signal_prediction.main import main
from fcs_signal_prediction.utils import data_utils as du


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

        result = main(exp_dir, exp_ref, low_control, high_control)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_ref', help='experimental reference from data science table')
    parser.add_argument('--exp_ref_dir', help='path to experimental reference directory')

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_exp_ref_dir = args.exp_ref_dir

    run_fcs_signal_prediction(arg_exp_ref, arg_exp_ref_dir)
