"""
run all parts of the Precomputed Data Table pipeline to gather, analyze, save data;
make summaries of the data, and make a record of the data product.

to see arguments and descriptions:
python run_pdt.py --help
example command:
python run_pdt.py "YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208" ../pdt_output/ hpc_path

:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import argparse
import os
import json
from datetime import datetime
import run_od_growth_analysis as run_growth


def get_latest_er(exp_ref, dc_dir):
    # must be looking in sd2e-projects/sd2e-project-43/reactor_outputs/complete
    all_er_dirs = os.listdir(dc_dir)
    if exp_ref not in all_er_dirs:
        no_er_dir_msg = 'There is no complete version of the Experimental Reference from Data Converge'
        return no_er_dir_msg
    else:
        timestamp_dir_list = os.listdir(os.path.join(dc_dir, exp_ref))
        latest_dir = sorted(timestamp_dir_list, reverse=True)[0]
        return latest_dir


def return_er_record_path(er_dir):

    record_json = 'record.json'
    files = os.listdir(er_dir)
    if record_json not in files:
        return 'Experimental Reference status can not be determined. No record.json file found'
    else:
        return os.path.join(er_dir, record_json)


def check_er_status(record_json):

    out_status = {}
    total_ok = 0

    with open(record_json, 'r') as j:
        record_status = json.load(j)['status']
        for data_type, data_status in record_status.items():
            if 'ok' in data_status:
                out_status[data_type] = data_status
                total_ok += 1
            else:
                out_status[data_type] = data_status

        if total_ok < len(record_status.keys()):
            return 'Data status NOT 100% ok:\n{}'.format(out_status)
        elif total_ok == len(record_status.keys()):
            return 'Data status 100% ok:\n{}'.format(out_status)


def confirm_data_types(er_file_list):

    dtype_confirm_dict = {'platereader': False, 'fc_raw_log10': False,
                          'fc_raw_events': False, 'fc_etl': False}

    for dtype in dtype_confirm_dict.keys():
        for file in er_file_list:
            if dtype in file:
                dtype_confirm_dict[dtype] = True

    return dtype_confirm_dict

# def main(exp_ref, out_dir, tacc_path_type, archive_system):
def main(exp_ref, out_dir, data_converge_dir):
    # make a new dir for output
    now = datetime.now()
    datetime_stamp = now.strftime('%Y%m%d%H%M%S')
    out_dir = os.path.abspath(out_dir)
    out_dir = os.path.join(out_dir, "pdt_{0:s}_{1:s}".format(exp_ref, datetime_stamp))
    print("making directory... ", out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # get latest data converge product for ER
    er_dir = get_latest_er(exp_ref, data_converge_dir)
    path_to_er_dir = os.path.join(data_converge_dir, exp_ref, er_dir)

    # Check status of data in ER's record.json file
    path_to_record_json = return_er_record_path(path_to_er_dir)
    check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    data_confirm_dict = confirm_data_types(os.listdir(path_to_er_dir))

    if data_confirm_dict['platereader']:
        run_growth.run_od_analysis(exp_ref, path_to_er_dir, data_confirm_dict, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_ref", help="experimental reference from data science table")
    parser.add_argument("output_dir", help="directory where to write the output files")
    parser.add_argument("data_converge_dir", help="path to Data Converge directory", )

    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_out_dir = args.output_dir
    arg_data_converge_dir = args.data_converge_dir

    main(arg_exp_ref, arg_out_dir, arg_data_converge_dir)
