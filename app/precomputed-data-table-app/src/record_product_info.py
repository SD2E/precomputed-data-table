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


def make_hash(file_path):
    """
    make a md5 hash for a file
    :param file_path:
    :return:
    """
    md5 = hashlib.md5()

    with open(file_path, 'rb') as file:
        # read in binary. chunks in case of big files
        while True:
            data = file.read(65536)
            if not data:
                break
            md5.update(data)

    hash = md5.hexdigest()
    print("MD5: {0}".format(hash))

    return hash


def make_hashes_for_files(out_dir, file_list):
    """
    for each file in a directory, make path, call hash, and make a record with file name and hash

    :param out_dir: path to output directory
    :param file_list: list of file records [{"name": x}, ...]
    :return: list of file records [{"name": str, "hash_md5": str}, ...]
    """

    for file in file_list:
        file_path = os.path.join(out_dir, file['name'])
        hash = make_hash(file_path)
        file['hash_md5'] = hash

    return file_list


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


def make_product_record(exp_ref, out_dir, files):
    datetime_stamp = datetime.now().strftime('%Y%m%d%H%M%S')

    upstream_status = get_comparison_passed(exp_ref) ## change to show data converge info
    version_info = get_dev_git_version()

    record = {
        "precomputed_data_table version": version_info,
        "experiment_reference": exp_ref,
        "date_run": datetime_stamp,
        "output_dir": out_dir,
        "files": files,
        "status_upstream": upstream_status
    }

    return record