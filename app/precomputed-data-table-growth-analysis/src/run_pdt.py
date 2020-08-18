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
import record_product_info as rpi


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


def main(exp_ref, analysis, out_dir, data_converge_dir):

    # Check status of data in ER's record.json file
    path_to_record_json = return_er_record_path(data_converge_dir)
    check_er_status(path_to_record_json)

    # confirm presence of data(frame) types
    data_confirm_dict = confirm_data_types(os.listdir(data_converge_dir))

    record_path = os.path.join(out_dir, "record.json")

    record = {}
    if analysis == 'xplan-od-growth-analysis':
        import run_od_growth_analysis as run_growth

        rg_od_analysis_df, pr_fname = run_growth.run_od_analysis(exp_ref, data_converge_dir, data_confirm_dict)
        od_growth_fname = 'pdt_{}__od_growth_analysis.csv'.format(exp_ref)
        rg_od_analysis_df.to_csv(od_growth_fname, index=False)
        od_growth_fname_dict = {pr_fname: [od_growth_fname]}
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
