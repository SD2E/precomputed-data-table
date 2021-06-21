"""
run perform metrics for PDT

:authors: anastasia deckard (anastasia.deckard@geomdata.com)
"""

import os

app_name = "perform-metrics-app"
app_id = "perform_metrics_app-0.1.1"

map_config_type = {
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

map_config_file = {
    'ys_annot': {'FLOW': 'annot_ind__fc_etl.json',
                 'PLATE_READER': 'annot_ind__pr.json'},
    'ys_1.0': {'FLOW': 'ys1_fc_etl.json',
               'PLATE_READER': 'ys1_pr.json'},
    'ys_ind': {'FLOW': 'general_ts_fc_etl_inducer.json',
               'PLATE_READER': 'general_ts_pr_inducer.json'},
}


def get_job_template(out_sys, out_dir, dc_batch_path, experiment_reference, mtype):
    """
    create a job template with info needed to run the perform metrics-app

    :param out_sys: 'sd2e-projects'
    :param out_dir: 'sd2e-project-48/complete/<exp_ref>/<datetime>'
    :param dc_batch_path: 'agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/<exp_ref>/<datetime>'
    :param experiment_reference: 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'
    :return:
    """

    job_template = None
    config_type = None
    config_file = None
    product_patterns = None

    if experiment_reference in map_config_type.keys():
        config_type = map_config_type[experiment_reference]
        if config_type in map_config_file.keys() and mtype in map_config_file[config_type].keys():
            config_file = map_config_file[config_type][mtype]

    if config_file:

        out_dir = os.path.join(out_dir, 'perform-metrics__' + mtype)

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
        if mtype == 'FLOW':
            perform_metrics_data = '{0:s}/{1:s}__fc_etl_stats.csv'.format(dc_batch_path, experiment_reference)
            perform_metrics_optional_data = '{0:s}/{1:s}__fc_meta.csv'.format(dc_batch_path, experiment_reference)
            job_template['inputs']['perform_metrics_data'] = perform_metrics_data
            job_template['inputs']['perform_metrics_optional_data'] = perform_metrics_optional_data
            data_files = [perform_metrics_data, perform_metrics_optional_data]

        elif mtype == 'PLATE_READER':
            perform_metrics_data = '{0:s}/{1:s}__platereader.csv'.format(dc_batch_path, experiment_reference)
            job_template['inputs']['perform_metrics_data'] = perform_metrics_data
            data_files = [perform_metrics_data]

        # figure out configuration files
        perform_metrics_config_path = "/perform_metrics/src/perform_metrics/configs/"
        job_template['parameters']['perform_metrics_config_json'] = os.path.join(perform_metrics_config_path,
                                                                                 config_file)

        product_patterns = [
            {'patterns': ['^.*(tsv)$'],
             'derived_from': data_files,
             'derived_using': []
             }]

    return job_template, product_patterns
