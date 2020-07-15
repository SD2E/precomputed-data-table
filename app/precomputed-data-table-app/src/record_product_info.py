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
    file_list_dict = []

    for file in file_list:
        file_path = os.path.join(out_dir, file)
        hash = make_hash(file_path)
        file_dict = {'name': file,
                     'hash_md5': hash}
        file_list_dict.append(file_dict)

    return file_list_dict

def append_record(record, files_dict, analysis, out_dir):

    # TODO add md5 check of dc files
    analysis_dict = {'files': []}

    for in_file in files_dict.keys():
        out_file_hashes = make_hashes_for_files(out_dir, files_dict[in_file])
        analysis_dict['files'].append({
            'data_converge input': in_file,
            'precomputed_data_table outputs': out_file_hashes
        })

    if 'analyses' not in record:
        record['analyses'] = {}
    record['analyses'][analysis] = analysis_dict
    return record
