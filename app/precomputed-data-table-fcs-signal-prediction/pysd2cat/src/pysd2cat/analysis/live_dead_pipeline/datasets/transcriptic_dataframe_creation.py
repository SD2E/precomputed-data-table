import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import expanduser
import requests
import zipfile
import shutil
import fnmatch
import glob
import ast
import logging
import transcriptic
from transcriptic import commands, Container
from transcriptic.config import Connection
from transcriptic.jupyter import objects
from pysd2cat.data import tx_fcs
from pysd2cat.data import pipeline
from pysd2cat.analysis import biofab_live_dead_analysis as blda
from pysd2cat.data import biofab_live_dead as bld

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    # ipython magic commands --> need to figure out what to do with them
    # %load_ext autoreload
    # %reload_ext autoreload
    # %autoreload 2

    run_id = 'r1dnj926jr7q9g'

    aliquot_map_technique = {
        'r1dd37mcxv5pf4': "titration1",
        'r1dk8xp9dymm54': "fifteen_zero",
        'r1dmsrursbqwuz': "10to80",
        'r1dnj926jr7q9g': "10to80"
    }

    work_dir = 'notebooks/data/transcriptic/' + run_id
    Connection.from_file("~/.transcriptic")
    tx_config = json.load(open(os.path.join(expanduser("~"), ".transcriptic")))

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='mylog.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)

    flow_data, plate_properties = bld.write_flow_data_and_metadata(run_id, tx_config, aliquot_map_technique[run_id], work_dir=work_dir)
    # flow_data.drop(columns=["Unnamed: 0"])

    print(plate_properties[0]["container"])

    df = bld.get_container_data(work_dir, plate_properties[0]['container']).drop(columns=["Unnamed: 0"])
    print(df)

    count = 0
    for p in plate_properties:
        df = bld.get_container_data(work_dir, p['container']).drop(columns=["Unnamed: 0"])
        time_point = flow_data.loc[flow_data['container_name'] == p['container']]['time_point'].iloc[0]
        df["time_point"] = time_point
        if count == 0:
            full_df = df.copy()
            count += 1
        else:
            full_df = full_df.append(df)
        print(full_df.shape)

    print(full_df)


if __name__ == '__main__':
    main()
