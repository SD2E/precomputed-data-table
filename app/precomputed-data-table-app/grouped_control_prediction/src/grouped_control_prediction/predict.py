from pandas import DataFrame
import pandas as pd
import numpy as np
import pickle
import os
from os.path import expanduser
import random
from typing import Optional, List
import pkg_resources

from pysd2cat.analysis import correctness


def do_analysis(df, experiment, controls, low_control, high_control, out_path):
    
    if df is None:
        print("Failed to get Data and Metadata")
        return None, None
    
    ## Setup parameters
    out_dir = out_path
    high_control = "WT-Live-Control"
    low_control = "WT-Dead-Control"
    controls['class_label'] = controls.apply(lambda x: 1 if x.strain_name == high_control else 0, axis=1)
    df.loc[:, 'output'] = 1 #assume alive
    strain_col = 'strain_name'  
    channels = ['FSC-A', 'SSC-A', 'BL1-A', 'FSC-W', 'FSC-H', 'SSC-W', 'SSC-H']
    
    # Run classification
    res = correctness.compute_predicted_output(df, 
                             training_df=controls,
                             data_columns = channels, 
                             out_dir=out_dir,
                             strain_col=strain_col,
                             high_control=high_control, 
                             low_control=low_control,
                             id_col="sample_id",
                             use_harness=True,
                             description=str(experiment)+"_live_dead")
    
    return (res, df)


def do_analysis_wrapper(df_original, experiment, controls, low_control, high_control, out_path):
    (res, df) = do_analysis(df_original, experiment, controls, low_control, high_control, out_path)
    return res, df


def idw_sample(distance_dict, num_samples):
    # Compute total distance of all controls
    total_dist = sum(list(distance_dict.values()))
    
    # Compute inverse distances
    inv_dists = []
    for d in list(distance_dict.values()):
        inv_dists.append((1/d)/total_dist)
    
    # Compute inverse weights
    total_inv_dist = sum(inv_dists)
    inv_weights = np.asarray(inv_dists)/total_inv_dist
    
    # Randomly sample controls without replacement using inverse weights
    return np.random.choice(list(distance_dict.keys()), num_samples, p=inv_weights, replace=False)


def top_n(distance_dict, num_samples):
    # Sort by dict value (Wasserstein distance) in ascending order
    distance_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])}
    # Return top n=num_samples control sets
    return list(distance_dict.keys())[:num_samples]


def predict_signal(df_original : DataFrame,
                   experiment: str,
                   project_id : str,
                   low_control : str,
                   high_control : str,
                   weighted : bool,
                   wass_path : str,
                   control_size : int,
                   id_col : str,
                   channels : List[str],
                   out_path : str,
                   strain_col : Optional[str]='strain_name') -> DataFrame:
    """
    Get predictions via grouped controls method.
    Sample n=control_size control experiments, group them together into one DataFrame,
    and use it as the training set for random forest classification.
    """
    
    # Extract control set Wasserstein distances
    distance_pkl = pkg_resources.resource_filename("grouped_control_prediction", wass_path)
    distances = pickle.load(open(distance_pkl, "rb" ))
    
    # Randomly sample n=control_size control sets based using inverse distance weighting
    if weighted:
        sampled_controls = top_n(distances, control_size)
    else:
        sampled_controls = random.sample(distances.keys(), k=control_size)
    
    # Compute average control distance
    avg_dist = 0
    for s in sampled_controls:
        avg_dist += distances[s]
    avg_dist /= control_size
    print("Average control distance: " + str(avg_dist))
    
    # Set up path to control data
    XPLAN_PROJECT = project_id
    xplan_base = os.path.join(expanduser("~"), 'sd2e-projects', XPLAN_PROJECT)
    path = os.path.join(xplan_base, 'xplan-reactor', 'data', 'transcriptic')
    
    # Iterate through sampled control set data and group them into one control/training set
    all_controls = pd.DataFrame()
    count = 0
    print("Grouping sampled control sets...")
    for sampled_exp in sampled_controls:
        
        # Extract sampled control set data
        experiment_flow = pd.read_csv(os.path.join(path, sampled_exp), index_col=0)
        controls = experiment_flow.loc[(experiment_flow.strain_name == "WT-Live-Control") | (experiment_flow.strain_name == "WT-Dead-Control")]
        channels_under = [x.replace('-', '_') for x in channels]
        renaming = dict(zip(channels_under, channels))
        controls = controls.rename(columns=renaming)

        # Add sampled control experiment to all_controls DataFrame
        count += 1
        print(str(count) + '/' + str(control_size) + '(' + sampled_exp + ')')
        all_controls = all_controls.append(controls, ignore_index=True)
    
    # Perform classification with new grouped control set
    res, df = do_analysis_wrapper(df_original, experiment, all_controls, low_control, high_control, out_path)
    
    # Store prediction set predcted output
    pred = pd.DataFrame()
    pred['predicted_output'] = res.predicted_output

    # Extract testing accuracy
    leader_board = pd.read_html(os.path.join(out_path, 'test_harness_results/custom_classification_leaderboard.html'))[0]
    leader_board = leader_board.sort_values(by=['Date', 'Time'], ascending=True)
    test_accuracy = leader_board.loc[:,'Accuracy'].iloc[-1]
    
    return pred, avg_dist, test_accuracy, all_controls
