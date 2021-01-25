"""
run diagnose for PDT

:authors: anastasia deckard (anastasia.deckard@geomdata.com), tessa johnson (tessa.johnson@geomdata.com)
"""

import os

app_name = "diagnose_app"
app_id = "diagnose_app-0.1.0"


def get_job_template(out_sys, out_dir, dc_batch_path, experiment_reference, mtype):
    """
    create a job template with info needed to run the diagnose-app

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

    config_file = 'pm_ys1_fc_etl.json'

    if config_file:
        out_dir = os.path.join(batch_dir, exp_ref)

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

        # figure out input files
        data_files = list()
        if mtype == 'FLOW':
            diagnose_exp_file = '{0:s}/per_sample_metric_circuit_media.tsv'.format(dc_batch_path)
            diagnose_optional_data = '{0:s}/{1:s}__fc_meta.csv'.format(dc_batch_path, experiment_reference)
            job_template['inputs']['exp_file'] = diagnose_exp_file
            job_template['inputs']['diagnose_optional_metadata'] = diagnose_optional_data
            data_files = [diagnose_exp_file, diagnose_optional_data]

        elif mtype == 'PLATE_READER':
            diagnose_exp_file = '{0:s}/per_sample_metric_circuit_media.tsv'.format(dc_batch_path)
            job_template['inputs']['exp_file'] = diagnose_exp_file
            data_files = [diagnose_exp_file]

            # figure out configuration files
            diagnose_config_path = "/diagnose/src/diagnose/configs/"
            job_template['parameters']['diagnose_config_json'] = os.path.join(diagnose_config_path, config_file)


    product_patterns = [
            {'patterns': ['^.*(tsv)$'],
             'derived_from': data_files,
             'derived_using': []
             }]

    return job_template, product_patterns
