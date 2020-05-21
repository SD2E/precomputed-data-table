"""
run xplan_od_growth_analysis on plate reader data frames produced by Data Converge;

example command:
python run_od_growth_analysis.py "YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208"

:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import pandas as pd
import argparse
import os
from xplan_od_growth_analysis import analysis_frame_api


def grab_pr_dataframe(exp_ref, er_dir):

    pr_file_name = '__'.join([exp_ref, 'platereader.csv'])
    platereader_df = pd.read_csv(os.path.join(er_dir, pr_file_name))

    return platereader_df


def grab_meta_dataframe(exp_ref, er_dir):

    meta_file_name = '__'.join([exp_ref, 'fc_meta.csv'])
    meta_df = pd.read_csv(os.path.join(er_dir, meta_file_name))

    return meta_df


def growth_analysis(platereader_df, exp_ref):

    od_analysis_df = analysis_frame_api.augment_dataframe(df=platereader_df, experiment_identifier=exp_ref)

    return od_analysis_df


def rows_to_replicate_groups(data_df, m_type):

    if m_type == 'od':
        # drop duplicates and induction samples
        # *** This will be replaced when replicate group column is added
        dup_cols = ['doubling_time', 'n0', 'inducer_type', 'well', 'experiment_id']
        data_drop_df = data_df.drop_duplicates(subset=dup_cols, keep='first')
        data_drop_df.dropna(subset=['inducer_type', 'inducer_concentration',
                                              'inducer_concentration_unit'], inplace=True)
        # drop sample-specific columns
        drop_cols = ['_id', 'sample_id', 'replicate', 'timepoint', 'timepoint_unit', 'container_id',
                     'aliquot_id', 'fluor_gain_0.16', 'fluor_gain_0.16/od']
        data_drop_df.drop(drop_cols, axis=1, inplace=True)
        # ***
        return data_drop_df

    elif m_type == 'fc':
        # *** This will be replaced when replicate group column is added
        # drop duplicates
        dup_cols = ['well_id', 'experiment_id']
        data_drop_df = data_df.drop_duplicates(subset=dup_cols, keep='first')
        # drop sample-specific columns
        drop_cols = ['_id', 'sample_id', 'replicate', 'timepoint', 'timepoint_unit', 'TX_plate_name']
        data_drop_df.drop(drop_cols, axis=1, inplace=True)
        # ***
        return data_drop_df


# def merge_growth_fc(growth_df, fc_meta_df):
#
#     growth_fc_df = fc_meta_df.merge(growth_df, )


def run_od_analysis(exp_ref, exp_ref_dir, conf_dict):

    pr_df = grab_pr_dataframe(exp_ref, exp_ref_dir)
    initial_od_analysis_df = growth_analysis(pr_df, exp_ref)
    # make df rows = replicate groups
    rg_od_analysis_df = rows_to_replicate_groups(initial_od_analysis_df, 'od')

    if conf_dict['fc_raw_log10']:
        fc_meta_df = grab_meta_dataframe(exp_ref, exp_ref_dir)
        rg_fc_meta_df = rows_to_replicate_groups(fc_meta_df, 'fc')

    elif not conf_dict['fc_raw_log10']:
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_ref', help='experimental reference from data science table')
    parser.add_argument('exp_ref_dir', help='path to experimental reference directory')
    parser.add_argument('data_confirm_dict', help='dictionary containing information on available data'
                                                  ' in experimental reference')
    # parser.add_argument("output_dir", help="directory where to write the output files")
    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_exp_ref_dir = args.exp_ref_dir
    arg_conf_dict = args.data_confirm_dict

    run_od_analysis(arg_exp_ref, arg_exp_ref_dir, arg_conf_dict)
