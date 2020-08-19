import sys
import pymongo
import numpy as np
import pandas as pd
from math import ceil
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt

# warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)

# Notes:
# catalog_staging = database, science_table = table in that database, many records inside of table.
# Every record in the science_table is metadata for an fcs file
# Each fcs file is for a single well

DB_URI = 'mongodb://readonly:WNCPXXd8ccpjs73zxyBV@catalog.sd2e.org:27020/admin?readPreference=primary'
EXP_IDS = [
    "experiment.transcriptic.r1eaf25ne8ajts",
    "experiment.transcriptic.r1eaf248xavu8a"
]


def query_catalog(experiment_ids=None):
    """
    Queries the TACC data catalog and returns a list of dictionaries containing relevant fcs file metadata
    :param experiment_ids: list of experiment ids. Defaults to EXP_IDS.
    :type experiment_ids: list
    :return:
    :rtype:
    """
    if experiment_ids is None:
        experiment_ids = EXP_IDS

    # Query science table
    client = pymongo.MongoClient(DB_URI)
    db = client.catalog_staging
    science_table = db.science_table
    query = {}
    # query for finding all records that have an experiment_id that matches an entry in the variable `experiment_ids`.
    query['experiment_id'] = {'$in': experiment_ids}

    # takes all of the elements in science_table that match the query and puts them in results list. Each element is a dictionary.
    results = []
    for match in science_table.find(query):
        results.append(match)
    # print(len(results))
    return results


def create_metadata_df(query_results, desired_keys=None, desired_sample_contents=None):
    """
    Creates a dataframe of metadata from the query results

    """
    # provide this a dictionary. Key is a column in the metadata dataframe,
    # and value is the name of the thing it should look for in sample_contents. Iterate through that dictionary
    # for sytox the key becomes stain and the value becomes sytox
    # if we want to combine dataframes, then you would want your key to be stain and value also stain
    # should collapse stain and ethanol parts into 1
    # need concentration too
    # easiest solution is probably always pull value and units

    if desired_keys is None:
        desired_keys = ["sample_id", "strain", "timepoint", "replicate", "filename", "jupyter_path"]
    if desired_sample_contents is None:
        desired_sample_contents = {
            "ethanol": ["label", "unit", "value"],
            "sytox": ["label"]
        }

    meta_data = {}

    # Todo: add try blocks
    for record in query_results:
        if 'flow' in str.lower(record['measurement_type']):
            meta_entry = {}
            for key in desired_keys:
                if isinstance(record[key], dict):
                    value = record[key]["value"]
                else:
                    value = record[key]
                meta_entry[key] = value

            sample_contents = record["sample_contents"]
            sample_contents_ids = [str.lower(id["name"]["label"]) for id in sample_contents]
            for dsc_key, dsc_values in desired_sample_contents.items():
                if dsc_key not in sample_contents_ids:
                    pass

            sytox = [content for content in record['sample_contents'] if 'sytox' in str.lower(content['name']['label'])]
            # print(sytox)
            if sytox:
                meta_entry['stain'] = 1
            else:
                meta_entry['stain'] = 0

            ethanol_content = [content for content in record['sample_contents']
                               if 'ethanol' in str.lower(content['name']['label'])]
            if not ethanol_content:
                ethanol = 0
                ethanol_unit = '%'
            if ethanol_content:
                try:
                    ethanol = ethanol_content[0]['value']
                    ethanol_unit = ethanol_content[0]['unit']
                except Exception as e:
                    print("No Ethanol concentration available for ", record['sample_id'])
                    ethanol = 0
                    ethanol_unit = '%'
            meta_entry['ethanol'] = ethanol
            meta_entry['ethanol_unit'] = ethanol_unit
            meta_entry['filename'] = record['filename']
            meta_entry['jupyter_path'] = record['jupyter_path']

            meta_data[record['sample_id']] = meta_entry

    # these are the keys of the dictionary that we are interested in
    cols = ['sample_id', 'time_point', 'strain',
            'replicate', 'stain',
            'ethanol_concentration', 'ethanol_unit', 'filename', 'jupyter_path']
    rows = []
    for sample_id, sample_metadata in meta_data.items():
        value_list = []
        for data_type, value in sample_metadata.items():
            value_list.append(str(value))
        row = []
        row = [sample_id] + value_list
        rows.append(row)
    df_metadata = pd.DataFrame(rows, columns=cols)
    return df_metadata


def create_flow_dataframe(metadata_df, strain="yeast", kill_method="ethanol"):
    """
    Takes in a metadata_df, and returns a dataframe of flow data based on the strain and kill_method chosen.
    :param strain:
    :type strain:
    :param kill_method:
    :type kill_method:
    :return:
    :rtype: DataFrame
    """
    pass
    # give it a set of filters through dictionary. Remove from metadata df the stuff that you don't want.
    # have option of combining dataframes (e.g. kill_method)


def main():
    query_results = query_catalog()
    metadata_df = create_metadata_df(query_results)

    # print(metadata_df)


if __name__ == '__main__':
    main()
