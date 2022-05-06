"""
Extracted from run_pdt.py
"""
import os
import json

# Note from George: This function is not used anywhere
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
