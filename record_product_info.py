"""
gather record product info and make data files summaries for this run of precomputed data table

:authors: robert c. moseley (robert.moseley@duke.edu) and  anastasia deckard (anastasia.deckard@geomdata.com)
"""

import hashlib
import os
import json
import pandas as pd

from datetime import datetime
# import pymongo
#
#
# def get_db_conn():
#     dbURI = 'mongodb://readonly:WNCPXXd8ccpjs73zxyBV@catalog.sd2e.org:27020/admin?readPreference=primary'
#     client = pymongo.MongoClient(dbURI)
#     db = client.catalog_staging
#
#     return db


def get_dev_git_version():
    import os
    stream = os.popen('git show -s --format="gitinfo: %h %ci"')
    output = stream.read().strip()
    if output.startswith('gitinfo:'):
        output = output.replace('gitinfo: ', '')
    else:
        output = "NA"

    version_info = output

    return version_info


def make_product_record(exp_ref, out_dir, dc_dir):

    datetime_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    version_info = get_dev_git_version()

    record = {
        "precomputed_data_table version": version_info,
        "experiment_reference": exp_ref,
        "date_run": datetime_stamp,
        "output_dir": out_dir,
        "analyses": {},
        "status_upstream": {
            'data-converge directory': dc_dir
        }
    }

    return record
