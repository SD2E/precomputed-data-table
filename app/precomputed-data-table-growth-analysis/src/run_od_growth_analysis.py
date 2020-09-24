"""
run xplan_od_growth_analysis on plate reader data frames produced by Data Converge;

example command:
python run_od_growth_analysis.py "YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208"

:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import pandas as pd
import argparse
import os
import json
from xplan_od_growth_analysis import analysis_frame_api
from common import record_product_info as rpi
from common import preproc


def grab_pr_dataframe(exp_ref, er_dir):

    pr_file_name = '__'.join([exp_ref, 'platereader.csv'])
    platereader_df = pd.read_csv(os.path.join(er_dir, pr_file_name))

    return platereader_df, pr_file_name


def grab_meta_dataframe(exp_ref, er_dir):

    meta_file_name = '__'.join([exp_ref, 'fc_meta.csv'])
    meta_df = pd.read_csv(os.path.join(er_dir, meta_file_name))

    return meta_df


def growth_analysis(platereader_df, exp_ref):
    return_dict: analysis_frame_api.AdReturn = analysis_frame_api.augment_dataframe(df=platereader_df, experiment_identifier=exp_ref)
    od_analysis_df = return_dict['od_df']
    assert isinstance(od_analysis_df, pd.DataFrame)

    return od_analysis_df


def rows_to_replicate_groups(data_df, m_type):

    if m_type == 'od':
        # drop duplicates and induction samples
        dup_cols = ['doubling_time', 'n0', 'inducer_type', 'well', 'experiment_id']
        data_drop_df = data_df.drop_duplicates(subset=dup_cols, keep='first')

        # drop sample-specific columns
        drop_cols = ['_id', 'sample_id', 'replicate', 'replicate_group', 'replicate_group_string',
                     'timepoint', 'timepoint_unit', 'container_id',
                     'aliquot_id']
        drop_cols += [col for col in data_drop_df.columns.tolist() if 'fluor_gain' in col]

        data_drop_df.drop(drop_cols, axis=1, inplace=True)

        return data_drop_df

    elif m_type == 'fc':
        # drop duplicates
        dup_cols = ['well', 'experiment_id']
        data_drop_df = data_df.drop_duplicates(subset=dup_cols, keep='first')
        # drop sample-specific columns
        drop_cols = ['_id', 'sample_id', 'replicate', 'replicate_group', 'replicate_group_string',
                     'timepoint', 'timepoint_unit', 'TX_plate_name']
        data_drop_df.drop(drop_cols, axis=1, inplace=True)
        # ***
        return data_drop_df


def run_od_analysis(exp_ref, exp_ref_dir, conf_dict):

    pr_df, pr_fname = grab_pr_dataframe(exp_ref, exp_ref_dir)
    od_analysis_initial_df = growth_analysis(pr_df, exp_ref)
    od_analysis_initial_df['experiment_id'] = od_analysis_initial_df['sample_id'].apply(lambda x: '.'.join(x.split('.')[-3:]))

    # make df rows = replicate groups
    rg_od_analysis_df = rows_to_replicate_groups(od_analysis_initial_df, 'od')

    return rg_od_analysis_df, pr_fname


def main(exp_ref, analysis, out_dir, data_converge_dir):

    # Check status of data in ER's record.json file
    path_to_record_json = preproc.return_er_record_path(data_converge_dir)
    preproc.check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    data_confirm_dict = preproc.confirm_data_types(os.listdir(data_converge_dir))

    record_path = os.path.join(out_dir, "record.json")

    rg_od_analysis_df, pr_fname = run_od_analysis(exp_ref, data_converge_dir, data_confirm_dict)
    od_growth_fname = 'pdt_{}__od_growth_analysis.csv'.format(exp_ref)
    rg_od_analysis_df.to_csv(od_growth_fname, index=False)
    od_growth_fname_dict = {pr_fname: [od_growth_fname]}
    record = {}
    record = rpi.append_record(record, od_growth_fname_dict, analysis, out_dir)

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
