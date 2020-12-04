"""
run xplan_od_growth_analysis on plate reader data frames produced by Data Converge;

example command:
python run_od_growth_analysis.py "YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208"

:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import os
import json
import argparse
import pandas as pd
from common import preproc
from typing import Optional, TypedDict
from pandas import DataFrame
from xplan_od_growth_analysis import credentials, analysis_frame_api
from common import record_product_info as rpi


def grab_pr_dataframe(exp_ref, er_dir):

    pr_file_name = '__'.join([exp_ref, 'platereader.csv'])
    platereader_df = pd.read_csv(os.path.join(er_dir, pr_file_name))

    return platereader_df, pr_file_name


def grab_meta_dataframe(exp_ref, er_dir):

    meta_file_name = '__'.join([exp_ref, 'fc_meta.csv'])
    meta_df = pd.read_csv(os.path.join(er_dir, meta_file_name))

    return meta_df, meta_file_name


def run_od_analysis(creds, platereader_df, exp_ref, fcs_meta_df: Optional[DataFrame] = None):

    if fcs_meta_df is None:
        return_dict: analysis_frame_api.AdReturn = analysis_frame_api.augment_dataframe(od_df=platereader_df,
                                                                                        experiment_identifier=exp_ref,
                                                                                        credentials_dict=creds)
        pr_analysis_df = return_dict['od_df']
        assert isinstance(pr_analysis_df, pd.DataFrame)
        return pr_analysis_df

    else:
        return_dict: analysis_frame_api.AdReturn = analysis_frame_api.augment_dataframe(od_df=platereader_df,
                                                                                        experiment_identifier=exp_ref,
                                                                                        fcs_meta_df=fcs_meta_df,
                                                                                        credentials_dict=creds)
        fcs_analysis_df = return_dict['fcs_meta_df']
        pr_analysis_df = return_dict['od_df']
        assert isinstance(pr_analysis_df, pd.DataFrame)
        assert isinstance(fcs_analysis_df, pd.DataFrame)
        return pr_analysis_df, fcs_analysis_df

def main(exp_ref, analysis, out_dir, data_converge_dir, arg_sift_ga_sbh_url, arg_sift_ga_sbh_user, arg_sift_ga_sbh_password, arg_sift_ga_mongo_user):

    # Check status of data in ER's record.json file
    path_to_record_json = preproc.return_er_record_path(data_converge_dir)
    preproc.check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    data_confirm_dict = preproc.confirm_data_types(os.listdir(data_converge_dir))

    record_path = os.path.join(out_dir, "record.json")

    pr_df, pr_fname = grab_pr_dataframe(exp_ref, data_converge_dir)
    pr_growth_fname = 'pdt_{}__pr_growth_analysis.csv'.format(exp_ref)

    creds: CredentialsDict = {
        'SIFT_GA_SBH_URL': arg_sift_ga_sbh_url,
        'SIFT_GA_SBH_USER': arg_sift_ga_sbh_user,
        'SIFT_GA_SBH_PASSWORD': arg_sift_ga_sbh_password,
        'SIFT_GA_MONGO_USER': arg_sift_ga_mongo_user
    }

    if data_confirm_dict['fc_raw_log10']:
        fcs_meta_df, fcs_meta_fname = grab_meta_dataframe(exp_ref, data_converge_dir)
        pr_analysis_df, fcs_analysis_df = run_od_analysis(creds, pr_df, exp_ref, fcs_meta_df)
        fcs_growth_fname = 'pdt_{}__fcs_growth_analysis.csv'.format(exp_ref)
    else:
        pr_analysis_df = run_od_analysis(creds, pr_df, exp_ref)

    pr_analysis_df.to_csv(pr_growth_fname, index=False)

    od_growth_fname_dict = {}
    od_growth_fname_dict[pr_fname] = [pr_growth_fname]

    if data_confirm_dict['fc_raw_log10']:
        fcs_analysis_df.to_csv(fcs_growth_fname, index=False)
        od_growth_fname_dict[fcs_meta_fname] = [fcs_growth_fname]

    record = {}
    record = rpi.append_record(record, od_growth_fname_dict, analysis, out_dir)

    with open(record_path, 'w') as jfile:
        json.dump(record, jfile, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_ref", help="experimental reference from data science table")
    parser.add_argument("--data_converge_dir", help="path to Data Converge directory")
    parser.add_argument("--analysis", help="analysis to run")
    parser.add_argument("--sift_ga_sbh_url", help="URL for the SynBioHub instance")
    parser.add_argument("--sift_ga_sbh_user", help="Username with access to the SynBioHub instance")
    parser.add_argument("--sift_ga_sbh_password", help="Password for `GA_SBH_USER` account")
    parser.add_argument("--sift_ga_mongo_user", help="Username with access to read experiment requests from MongoDB")

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_data_converge_dir = args.data_converge_dir
    arg_analysis = args.analysis
    arg_out_dir = "."
    arg_sift_ga_sbh_url = args.sift_ga_sbh_url
    arg_sift_ga_sbh_user = args.sift_ga_sbh_user
    arg_sift_ga_sbh_password = args.sift_ga_sbh_password
    arg_sift_ga_mongo_user = args.sift_ga_mongo_user

    main(arg_exp_ref, arg_analysis, arg_out_dir, arg_data_converge_dir, arg_sift_ga_sbh_url, arg_sift_ga_sbh_user, arg_sift_ga_sbh_password, arg_sift_ga_mongo_user)
