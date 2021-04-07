#!/usr/bin/env python
"""
:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
from scipy.stats import wasserstein_distance as wd

from grouped_control_prediction.utils import data_utils as du


def k_nearest_controls(experiment_df, control_set_dir):

    # Load prediction data
    # experiment_df = du.get_data_and_metadata(exp_dir)
    # experiment_df = du.get_data_and_metadata_mem(exp_dir)
    # code above ^^ was for loading the experiment data in. However, this function now takes in a pandas dataframe

    # Create renaming dict for channel columns
    channels = ['FSC-A', 'SSC-A', 'BL1-A', 'FSC-W', 'FSC-H', 'SSC-W', 'SSC-H']
    channels_under = [x.replace('-', '_') for x in channels]
    renaming = dict(zip(channels_under, channels))

    # Group prediction data
    exp_grouped = experiment_df.groupby(['strain_name', 'timepoint', 'inducer_concentration', 'replicate'])

    # Set path to control data
    # XPLAN_PROJECT = "sd2e-project-14"
    # xplan_base = os.path.join('/work/projects/SD2E-Community/prod/projects', XPLAN_PROJECT)
    # # xplan_base = os.path.join(expanduser("~"), 'sd2e-projects', XPLAN_PROJECT)
    # path = os.path.join(xplan_base, 'xplan-reactor', 'data', 'transcriptic')

    # Initialize control distance dictionary and counter
    distances = dict()
    count = 0

    # Set k value
    k = 10

    # Loop through control data files and get k-nearest using Wasserstein distance as 'closeness' metric
    for exp in os.listdir(control_set_dir):
        if ".csv" in exp:
            # Print counter
            count += 1
            print(str(count) + "/109 (" + exp + ")")

            # Extract current control data and rename channel columns
            exp_flow = pd.read_csv(os.path.join(control_set_dir, exp), index_col=0, dtype=str)
            exp_flow = exp_flow.astype({x:float for x in channels_under})
            controls = exp_flow.loc[
                (exp_flow.strain_name == "WT-Live-Control") | (exp_flow.strain_name == "WT-Dead-Control")]
            controls = controls.rename(columns=renaming)

            # Separate live/dead control data
            dead_controls = controls.loc[controls['strain_name'] == "WT-Dead-Control"]
            live_controls = controls.loc[controls['strain_name'] == "WT-Live-Control"]

            # Apply log-10 transform to channel columns
            dc_d = dead_controls[channels].apply(np.log10).replace([np.inf, -np.inf], 0.0)
            lc_d = live_controls[channels].apply(np.log10).replace([np.inf, -np.inf], 0.0)

            # Total Wasserstein distance of current control will be sum of distances of each sample to nearest control
            wass_dist = 0

            # Loop through samples
            for sample_name, sample_data in exp_grouped:
                #print("sample_name: {} sample_data.columns: {}".format(sample_name, sample_data.columns))
                if len(distances) == k and wass_dist >= max(distances.values()):
                    break

                # Apply log-10 transform to channel columns
                sample_data = sample_data.astype({x:float for x in channels})
                s_d = sample_data[channels].apply(np.log10).replace([np.inf, -np.inf], 0.0)

                # Get Wasserstein distance between each channel of sample to both live/dead controls
                dw = pd.Series()
                lw = pd.Series()
                #print("channels: {}".format(channels))
                for channel in channels:
                    #print("channel: {}".format(channel))
                    #print("dc_d[channel].dropna().size: {} lw[channel].dropna().size: {} s_d[channel].dropna().size: {}".format(dc_d[channel].dropna().size, lc_d[channel].dropna().size, s_d[channel].dropna().size))
                    dw[channel] = wd(dc_d[channel].dropna(), s_d[channel].dropna())
                    lw[channel] = wd(lc_d[channel].dropna(), s_d[channel].dropna())

                # Average over channels and add nearest control to overall distance
                if dw.mean() < lw.mean():
                    wass_dist += dw.mean()
                else:
                    wass_dist += lw.mean()

            # If there are already k nearest controls
            if len(distances) == k:
                # If the current wass_dist is not in the top k, continue
                if wass_dist >= max(distances.values()):
                    continue

                # If the current wass_dist is in the top k, replace it with the furthest
                else:
                    distances.pop(max(distances, key=distances.get))
                    distances[exp] = wass_dist
                    assert len(distances) == k

            # If we still haven't added k controls, keep adding
            else:
                distances[exp] = wass_dist
                assert len(distances) <= k

    return distances
