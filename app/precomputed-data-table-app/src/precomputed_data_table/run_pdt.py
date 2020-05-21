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
import precomputed_data_table.run_od_growth_analysis as run_growth


def get_latest_er(exp_ref, dc_dir):

    all_er_dirs = os.listdir(dc_dir)
    latest_dirs_dict = {x.split('_')[1]: x for x in sorted(all_er_dirs)}

    if exp_ref in latest_dirs_dict.keys():
        exp_ref_dir = latest_dirs_dict[exp_ref]
        return exp_ref_dir
    else:
        no_er_dir_msg = 'Experimental Reference not in Data Converge'
        return no_er_dir_msg


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
def main(exp_ref, out_dir):
    # make a new dir for output
    now = datetime.now()
    datetime_stamp = now.strftime('%Y%m%d%H%M%S')
    out_dir = os.path.abspath(out_dir)
    out_dir = os.path.join(out_dir, "dc_{0:s}_{1:s}".format(exp_ref, datetime_stamp))
    print("making directory... ", out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # data_converge_dir = '/home/jupyter/sd2e-projects/sd2e-project-43/test'
    # testing
    data_converge_dir = '/Users/robertmoseley/Desktop/SD2/sd2e_git/apps/precomputed-data-table/app/precomputed-data-table-app/tests/data/good_er_dirs'

    # get latest data converge product for ER
    er_dir = get_latest_er(exp_ref, data_converge_dir)
    path_to_er_dir = os.path.join(data_converge_dir, er_dir)

    # Check status of data in ER's record.json file
    path_to_record_json = return_er_record_path(path_to_er_dir)
    check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    data_confirm_dict = confirm_data_types(os.listdir(path_to_er_dir))

    if data_confirm_dict['platereader']:
        run_growth.run_od_analysis(exp_ref, path_to_er_dir, data_confirm_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_ref", help="experimental reference from data science table")
    parser.add_argument("output_dir", help="directory where to write the output files")
    # parser.add_argument("tacc_path_type", help="either 'hpc_path' or 'jupyter_path'", )
    # parser.add_argument("-a", "--archive_system", help="tacc archive system, ex data-sd2e-projects.sd2e-project-45")
    args = parser.parse_args()
    arg_exp_ref = args.experiment_ref
    arg_out_dir = args.output_dir
    # arg_tacc_path_type = args.tacc_path_type
    # arg_archive_system = args.archive_system

    # main(arg_exp_ref, arg_out_dir, arg_tacc_path_type, arg_archive_system)
    main(arg_exp_ref, arg_out_dir)
