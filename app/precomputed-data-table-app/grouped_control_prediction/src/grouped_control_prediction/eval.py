import argparse
import os
from os.path import expanduser
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Optional, List
from grouped_control_prediction.predict import predict_signal


parser = argparse.ArgumentParser(description='Predict Signal Output for samples')
parser.add_argument('data_converge_path', type=str,
                    help='Path to data converge results')
parser.add_argument('df', type=str,
                    help='Prediction dataframe')
parser.add_argument('meta', type=str,
                    help='Prediction meta dataframe')
parser.add_argument('project_id', type=str,
                    help='Path to control data')
parser.add_argument('low_control', type=str,
                    help='Strain Name of Low Control')
parser.add_argument('high_control', type=str,
                    help='Strain Name of High Control')
parser.add_argument('weighted', type=bool,
                    help='Use inverse distance weighted sampling for controls')
parser.add_argument('wass_path', type=str,
                    help='Path to Wasserstein Distances')
parser.add_argument('control_size', type=int,
                    help='Number of control experiments to group in training set')
parser.add_argument('--id_col', type=str,
                    help='Sample id column name', default="sample_id")
parser.add_argument('--strain_col', type=str,
                    help='Strain column name', default="strain_name")
parser.add_argument('--out_path', type=str,
                    help='Test Harness output location', default=".")


def main(data_converge_path : str,
         df : DataFrame,
         meta : DataFrame,
         project_id : str,
         low_control : str,
         high_control : str,
         weighted : bool,
         wass_path : str,
         control_size : int,
         id_col : Optional[str]="sample_id",
         strain_col : Optional[str]='strain_name',
         out_path : Optional[str]='.'):
    """
    Get the raw flow cytometry data, predict the output signal for each event,
    aggregate by sample, return metadata with columns for mean and std dev. of
    the predicted signal for each sample.
    """
    
    channels = ['FSC-A', 'SSC-A', 'BL1-A', 'FSC-W', 'FSC-H', 'SSC-W', 'SSC-H']
    
    # Predict the output signal for each event
    pred, avg_dist, test_accuracy, all_controls = predict_signal(df,
                                                                 data_converge_path,
                                                                 project_id,
                                                                 low_control,
                                                                 high_control,
                                                                 weighted,
                                                                 wass_path,
                                                                 control_size,
                                                                 id_col,
                                                                 channels,
                                                                 out_path,
                                                                 strain_col=strain_col)
    
    # Attach predictions to original data
    df['predicted_output'] = pred['predicted_output']
    
    # Get the mean and std output signal for each sample
    mean_prediction = df.groupby([id_col]).agg({"predicted_output" : [np.mean, np.std]}).reset_index()
    mean_prediction.columns  = mean_prediction.columns.map('_'.join)
    mean_prediction = mean_prediction.rename(columns={id_col+"_": id_col})
    
    # Attach mean & standard deviation of sample predcitions to the metadata
    result = meta.merge(mean_prediction, on=id_col)
    
    # Read in optical density (OD) data
    OD_DATA_CONVERGE_PROJECT = "sd2e-project-48"
    od_data_converge_base = os.path.join(expanduser("~"), 'sd2e-projects', OD_DATA_CONVERGE_PROJECT)
    od_experiment = os.path.join(od_data_converge_base, )
    od_experiment = os.path.realpath(os.path.join(od_data_converge_base, 'complete',
                                                  'YeastSTATES-CRISPR-Short-Duration-Time-Series-35C','20200625173644',
                                                  'xplan-od-growth-analysis'))
    od_df = pd.read_csv(os.path.join(od_experiment,
                                     'pdt_YeastSTATES-CRISPR-Short-Duration-Time-Series-35C__od_growth_analysis.csv'))
    
    od_df = od_df[od_df['inducer_type'] == 'beta-estradiol']
    # Store both sets of predictions as integers (0/1 -> dead/live)
    rf_preds = result.groupby(['experiment_id','well_id']).mean().predicted_output_mean.values
    od_preds = np.invert(od_df.dead).astype(int).values
    num_preds = len(od_preds)

    # Predictions compared to OD labels
    od_loss = sum(abs(od_preds-rf_preds))/num_preds
    od_accuracy = sum(np.around(rf_preds) == od_preds)/num_preds
    print("OD loss: " + str(od_loss))
    print("OD accuracy: " + str(od_accuracy))
    
    return result, avg_dist, test_accuracy, od_loss, od_accuracy
    


if __name__ == '__main__':
    args = parser.parse_args()
    
    data_converge_path = args.data_converge_path
    df = args.df
    meta = args.meta
    project_id = args.project_id
    low_control = args.low_control
    high_control = args.high_control
    weighted = args.weighted
    wass_path = args.wass_path
    control_size = args.control_size
    id_col = args.id_col
    strain_col = args.strain_col
    out_path = args.out_path
    
    main(data_converge_path,
         df,
         meta,
         project_id,
         low_control,
         high_control,
         weighted,
         wass_path,
         control_size,
         id_col=id_col,
         strain_col=strain_col,
         out_path=out_path)
