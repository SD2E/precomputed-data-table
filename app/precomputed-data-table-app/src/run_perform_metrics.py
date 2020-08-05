"""
run perform metrics for PDT

:authors: anastasia deckard (anastasia.deckard@geomdata.com)
"""

import os
from datetime import datetime
import json
from agavepy import Agave

ag = Agave.restore()

# TODO: this only supports plate reader right now, so should only be called from run_pdt where
#  dtype_confirm_dict['platereader'] == True

def main(out_sys, out_dir, dc_batch_path, experiment_reference):

    # if there are known experiment types or experiments refs that don't work
    # with perform-metrics, should probably exclude them here?


    # make a directory for perform_metrics output out_sys/out_dir/perform-metrics
    pm_dir = os.path.join(out_dir, 'perform-metrics')
    print("batch dir: ", pm_dir)

    # build a path to the file we want from the DC output; here we will use plate reader
    # use a config file for platereader data we've set up to work with these experiment refs,
    # these must be in the configs folder that is deployed with the code
    perform_metrics_data = '{0:s}/{1:s}__platereader.csv'.format(dc_batch_path, experiment_reference)
    perform_metrics_config_json = "/perform_metrics/src/perform_metrics/configs/metrics_y4d_ts_pr_inducer.json"

    job_template = {"name": "perform-metrics",
                    "appId": "perform_metrics-0.1.0",
                    "archive": True,
                    "archiveSystem": out_sys,
                    "archivePath": pm_dir,
                    "archiveOnAppError": True,
                    "maxRunTime": "12:00:00",
                    "inputs": {

                        "perform_metrics_data": perform_metrics_data
                    },
                    "parameters": {
                        "perform_metrics_config_json": perform_metrics_config_json
                    }
                    }

    job_id = ag.jobs.submit(body=job_template)['id']
    print(batch_dir)
    print(exp_ref, job_id)
    print(json.dumps(job_template, indent=4))


if __name__ == '__main__':
    # in prod, paths would be passed to us like:
    # out_sys = 'sd2e-projects'
    # out_dir = 'sd2e-project-48/complete/YeastSTATES-CRISPR-Growth-Curves-with-Plate-Reader-Optimization/20200722190009'
    # dc_batch_path = 'agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208/20200610192131'
    # experiment_reference = 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'

    # for testing to our local directories:
    out_sys = 'data-tacc-work-adeckard'
    out_dir = 'perform-metrics-uat'
    dc_batch_path = 'agave://data-sd2e-projects.sd2e-project-43/test/batch_20200713124857_master/YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'
    experiment_reference = 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'

    main(out_sys, out_dir, dc_batch_path, experiment_reference)