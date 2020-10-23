import argparse
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from grouped_control_prediction.predict import predict_signal
from grouped_control_prediction.utils import data_utils as du
from grouped_control_prediction.utils import plot


parser = argparse.ArgumentParser(description='Predict Signal Output for samples')
parser.add_argument('data_converge_path', type=str,
                    help='Path to data converge results')
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
parser.add_argument('control_set_dir', type=str, default='',
                    help='Path to control experiments for computing wasserstein distances \
                     instead of using cached distances.')
parser.add_argument('--id_col', type=str,
                    help='Sample id column name', default="sample_id")
parser.add_argument('--strain_col', type=str,
                    help='Strain column name', default="strain_name")
parser.add_argument('--out_path', type=str,
                    help='Test Harness output location', default=".")
parser.add_argument('--plot', type=bool, default=True,
                    help='Plot results')


def main(data_converge_path: str,
         project_id : str,
         low_control : str,
         high_control : str,
         weighted : bool,
         wass_path : str,
         control_size : int,
         control_set_dir: Optional[str] = '',
         id_col : Optional[str] = "sample_id",
         strain_col : Optional[str] = 'strain_name',
         out_path : Optional[str] = '.',
         plot : Optional[bool] = True):
    """
    Get the raw flow cytometry data, predict the output signal for each event,
    aggregate by sample, return metadata with columns for mean and std dev. of
    the predicted signal for each sample.
    """
    
    # Get the data and metadata
    print("Loading prediction data...")
    df = du.get_data_and_metadata(data_converge_path)
    meta = du.get_meta(data_converge_path, du.get_record(data_converge_path))
    # meta = meta.rename(columns={'well':'well_id'})
    channels = ['FSC-A', 'SSC-A', 'BL1-A', 'FSC-W', 'FSC-H', 'SSC-W', 'SSC-H']

    # Predict the output signal for each event from computed distances
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
                                                                 control_set_dir=control_set_dir,
                                                                 strain_col=strain_col)


    # Attach predictions to original data
    df['predicted_output'] = pred['predicted_output']
    
    # Get the mean and std output signal for each sample
    mean_prediction = df.groupby([id_col]).agg({"predicted_output" : [np.mean, np.std]}).reset_index()
    mean_prediction.columns = mean_prediction.columns.map('_'.join)
    mean_prediction = mean_prediction.rename(columns={id_col+"_": id_col})
    
    # Attach mean & standard deviation of sample predcitions to the metadata
    result = meta.merge(mean_prediction, on=id_col)
    
    # Drop Media Controls from RF data
    rf = result[~result['standard_type'].str.contains('BEAD')]
    # Trim experiment_id values
    shift = len('experiment.transcriptic.')
    rf['experiment_id'] = rf['experiment_id'].str[shift:]
    # Get aggregate timepoint mean/std predictions by grouping over experiment_id and well
    rf_df = rf.groupby(['experiment_id','well']).agg({'predicted_output_mean': [np.mean, np.std]})

    if plot == True:
        # Create plots
        well_timeseries = result.groupby(['timepoint', 'well', 'experiment_id']).agg(np.mean).sort_values(by=['well', 'timepoint', 'experiment_id']).reset_index()
        well_timeseries_fig = plot.plot_well_timeseries(well_timeseries)
        samples_and_controls_fig = plot.plot_samples_and_controls(df[[id_col, 'SSC-A']].merge(meta[[strain_col, id_col, "well", "timepoint", 'experiment_id']], on=id_col), result, low_control, high_control, 'SSC-A', all_controls[['SSC-A', strain_col]], 10000)
        return result, rf_df, test_accuracy, well_timeseries_fig, samples_and_controls_fig
    else:
        return result, rf_df, test_accuracy

if __name__ == '__main__':
    args = parser.parse_args()

    data_converge_path = args.data_converge_path
    project_id = args.project_id
    low_control = args.low_control
    high_control = args.high_control
    weighted = args.weighted
    wass_path = args.wass_path
    control_size = args.control_size
    control_set_dir = args.control_set_dir
    id_col = args.id_col
    strain_col = args.strain_col
    out_path = args.out_path
    plot = args.plot
    
    main(data_converge_path,
         project_id,
         low_control,
         high_control,
         weighted,
         wass_path,
         control_size,
         control_set_dir=control_set_dir,
         id_col=id_col,
         strain_col=strain_col,
         out_path=out_path,
         plot=plot)
