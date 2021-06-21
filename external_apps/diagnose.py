"""
run diagnose for PDT

:authors: anastasia deckard (anastasia.deckard@geomdata.com), tessa johnson (tessa.johnson@geomdata.com)
"""

import os

app_name = "diagnose_app"
app_id = "diagnose_app-0.1.1"

map_data_type = {
    'YeastSTATES-Activator-Circuit-BE-Short-Duration-Time-Series-30C': 'ys_annot',
    'YeastSTATES-Activator-Circuit-Dox-Short-Duration-Time-Series-30C': 'ys_annot',
    'YeastSTATES-Beta-Estradiol-OR-Gate-Plant-TF-Dose-Response': 'ys_annot',
    'YeastSTATES-Doxycycline-OR-Gate-Plant-TF-Dose-Response': 'ys_annot',
    'YeastSTATES-Dual-Response-CRISPR-Short-Duration-Time-Series-30C': 'ys_annot',
    'YeastSTATES-OR-Gate-CRISPR-Dose-Response': 'ys_annot',
    'YeastSTATES-Dual-Response-CRISPR-Redesigns-Short-Duration-Time-Series-30C-Titration': 'ys_annot',

    'YeastSTATES-1-0-Time-Series-Round-1': 'ys_1.0',
    'YeastSTATES-1-0-Time-Series-Round-1-1': 'ys_1.0',
    'YeastSTATES-1-0-Time-Series-Round-2-0': 'ys_1.0',
    'YeastSTATES-1-0-Time-Series-Round-3-0': 'ys_1.0',
    'YeastSTATES-1-0-Time-Series-Round-4-0': 'ys_1.0',

    'YeastSTATES-CRISPR-Dose-Response': 'ys_ind',
    'YeastSTATES-CRISPR-Long-Duration-Time-Series-20191208': 'ys_ind',
    'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208': 'ys_ind',
    'YeastSTATES-CRISPR-Short-Duration-Time-Series-35C': 'ys_ind',
}

map_data_file = {
    'ys_annot': 'per_sample_metric_str_cir_med_ind_tmp_tim.tsv',
    'ys_1.0': 'per_sample_metric_cir_med_tim.tsv',
    'ys_ind': 'per_sample_metric_str_med_ind_tmp_tim.tsv',
    'default': 'per_sample_metric_str_tim.tsv',
}


def get_job_template(out_sys, out_dir, pm_batch_path, experiment_reference, mtype):
    """
    create a job template with info needed to run the diagnose-app

    :param out_sys: 'sd2e-projects'
    :param out_dir: 'sd2e-project-48/complete/<exp_ref>/<datetime>'
    :param pm_batch_path: 'agave://data-sd2e-projects.sd2e-project-48/complete/<exp_ref>/<datetime>'
    :param experiment_reference: 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'
    :param mtype: 'FLOW', 'PLATE_READER'
    :return:
    """

    job_template = None
    data_type = None
    data_file = None
    product_patterns = None

    if experiment_reference in map_data_type.keys():
        data_type = map_data_type[experiment_reference]
    else:
        data_type = 'default'

    data_file = map_data_file[data_type]

    # currently only supporting flow etl; will add check for all data types here
    if mtype == 'FLOW' or mtype == "PLATE_READER":
        out_dir = os.path.join(out_dir, 'diagnose__' + mtype)

        job_template = {"name": app_name,
                        "appId": app_id,
                        "archive": True,
                        "archiveSystem": out_sys,
                        "archivePath": out_dir,
                        "archiveOnAppError": True,
                        "maxRunTime": "5:00:00",
                        "inputs": {},
                        "parameters": {}
                        }

        # figure out input files
        data_files = list()
        base_in_dir = os.path.join(pm_batch_path, 'perform-metrics__{0:s}'.format(mtype))
        diagnose_config_path = "/diagnose/src/diagnose/configs/"
        if mtype == 'FLOW':
            diagnose_exp_file = os.path.join(base_in_dir, data_file)
            diagnose_optional_data = os.path.join(base_in_dir,
                                                  '{0:s}__fc_meta.csv'.format(experiment_reference))
            job_template['inputs']['exp_file'] = diagnose_exp_file
            job_template['inputs']['diagnose_optional_metadata'] = diagnose_optional_data
            data_files = [diagnose_exp_file, diagnose_optional_data]

        if mtype == 'PLATE_READER':
            diagnose_exp_file = os.path.join(base_in_dir, data_file)
            diagnose_optional_data = os.path.join(base_in_dir,
                                                  '{0:s}__platereader.csv'.format(experiment_reference))
            job_template['inputs']['exp_file'] = diagnose_exp_file
            job_template['inputs']['diagnose_optional_metadata'] = diagnose_optional_data
            data_files = [diagnose_exp_file, diagnose_optional_data]

        # figure out configuration files
        config_file = 'diagnose_pm_general.json'
        job_template['parameters']['diagnose_config_json'] = os.path.join(diagnose_config_path, config_file)

        product_patterns = [
            {'patterns': ['^.*(tsv)$'],
             'derived_from': data_files,
             'derived_using': []
             }]

    return job_template, product_patterns


if __name__ == '__main__':
    out_sys = 'sd2e-projects'
    out_dir = 'sd2e-project-48/complete/<exp_ref>/<datetime>'
    pm_batch_path = 'agave://data-sd2e-projects.sd2e-project-48/complete/<exp_ref>/<datetime>'
    experiment_reference = 'YeastSTATES-1-0-Time-Series-Round-3-0'
    mtype = 'FLOW'

    job = get_job_template(out_sys, out_dir, pm_batch_path, experiment_reference, mtype)

    print(job)
