"""
run perform metrics for PDT

:authors: anastasia deckard (anastasia.deckard@geomdata.com)
"""

import os


# right now perform metrics can run on these supported types
#supported_cps = ['YEAST_STATES']
#supported_protocols = ['TimeSeriesHTP']
#supported_measurement_types = ['PLATE_READER']


#def is_supported(challenge_problem, protocol, measurement_type):
#    if challenge_problem not in supported_cps:
#        return False
#    if protocol not in supported_protocols:
#        return False
#    if measurement_type not in supported_measurement_types:
#        return False
#
#    return True


def get_job_template(out_sys, out_dir, dc_batch_path, experiment_reference, mtype):
    """
    create a job template with info needed to run the perform metrics-app
    currently just supporting one config to test it

    :param out_sys: 'sd2e-projects'
    :param out_dir: 'sd2e-project-48/complete/<exp_ref>/<datetime>'
    :param dc_batch_path: 'agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/<exp_ref>/<datetime>'
    :param experiment_reference: 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'
    :return:
    """

    # directory for perform_metrics output out_sys/out_dir/perform-metrics
    pm_dir = os.path.join(out_dir, 'perform-metrics')
    print("batch dir: ", pm_dir)

    # build a path to the file we want from the DC output; here we will use plate reader
    # use a config file for platereader from configs folder that is deployed with the code
    perform_metrics_data = '{0:s}/{1:s}__platereader.csv'.format(dc_batch_path, experiment_reference)
    perform_metrics_config_json = "/perform_metrics/src/perform_metrics/configs/metrics_y4d_ts_pr_inducer.json"

    job_template = {"name": "perform-metrics",
                    "appId": "perform_metrics_app-0.1.0",
                    "archive": True,
                    "archiveSystem": out_sys,
                    "archivePath": pm_dir,
                    "archiveOnAppError": True,
                    "maxRunTime": "2:00:00",
                    "inputs": {

                        "perform_metrics_data": perform_metrics_data
                    },
                    "parameters": {
                        "perform_metrics_config_json": perform_metrics_config_json
                    }
                    }

    product_patterns = [
        {'patterns': ['^.*(tsv)$'],
         'derived_from': [perform_metrics_data],
         'derived_using': []
         }]

    return job_template, product_patterns


if __name__ == '__main__':
    # an example of how to call this code:
    arg_challenge_problem = 'YEAST_STATES'
    arg_protocol = 'TimeSeriesHTP'
    arg_measurement_type = 'PLATE_READER'
    if is_supported(arg_challenge_problem, arg_protocol, arg_measurement_type):
        arg_out_sys = 'sd2e-projects'
        arg_out_dir = 'sd2e-project-48/complete/YeastSTATES-CRISPR-Growth-Curves-with-Plate-Reader-Optimization/20200722190009'
        arg_dc_batch_path = 'agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208/20200610192131'
        arg_experiment_reference = 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'
        my_job_template, product_patterns = get_job_template(arg_out_sys, arg_out_dir, arg_dc_batch_path, arg_experiment_reference)

        print('finished')