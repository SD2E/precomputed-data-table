import argparse
import numpy as np
from typing import Optional, List
from fcs_signal_prediction.predict import predict_signal
from fcs_signal_prediction.utils import data_utils as du
from fcs_signal_prediction.utils import plot



parser = argparse.ArgumentParser(description='Predict Signal Output for samples')
parser.add_argument('data_converge_path', type=str,
                    help='Path to data converge results')
parser.add_argument('experiment_id', type=str,
                    help='Experiment Identifier')
parser.add_argument('low_control', type=str,
                    help='Strain Name of Low Control')
parser.add_argument('high_control', type=str,
                    help='Strain Name of High Control')
parser.add_argument('--id_col', type=str,
                    help='Sample id column name', default="sample_id")
parser.add_argument('--strain_col', type=str,
                    help='Strain column name', default="strain_name")





def main(data_converge_path: str, 
         experiment_identifier: str,
         low_control : str,
         high_control : str,
         id_col : Optional[str]="sample_id",
         strain_col : Optional[str]='strain_name'):
    """
    Get the raw flow cytometry data, predict the output signal for each event,
    aggregate by sample, return metadata with columns for mean and std dev. of
    the predicted signal for each sample.
    """
    
    
    ## Get the data and metadata
    
    df = du.get_data(data_converge_path, du.get_record(data_converge_path))
    meta = du.get_meta(data_converge_path, du.get_record(data_converge_path))
    df = df.merge(meta[[id_col, strain_col]])
    
    
    ## Get the channels in the data
    
    channels = list(df.columns)
    channels.remove(id_col)
    channels.remove(strain_col)
    
    
    ## Predict the output signal for each event
    
    pred = predict_signal(df, 
                          experiment_identifier,
                          low_control,
                          high_control,
                          id_col,
                          channels,
                          strain_col=strain_col)

    
    ## Get the mean and std output signal for each sample
    
    mean_prediction = pred.groupby([id_col]).agg({"predicted_output" : [np.mean, np.std]}).reset_index()
    mean_prediction.columns  = mean_prediction.columns.map('_'.join)
    mean_prediction = mean_prediction.rename(columns={id_col+"_": id_col})

    
    ## Attach the mean and std to the metadata
    
    result = meta.merge(mean_prediction, on=id_col)
    
    
    well_timeseries = result.groupby(['timepoint', 'well_id', 'experiment_id']).agg(np.mean).sort_values(by=['well_id', 'timepoint', 'experiment_id']).reset_index()
    well_timeseries_fig = plot.plot_well_timeseries(well_timeseries)
    
    samples_and_controls_fig = plot.plot_samples_and_controls(df[[id_col, 'BL1-A']].merge(meta[[strain_col, id_col, "well_id", "timepoint", 'experiment_id']], on=id_col), result, low_control, high_control)
    
    return result, well_timeseries_fig, samples_and_controls_fig



if __name__ == '__main__':
    args = parser.parse_args()

    data_converge_path = args.data_converge_path
    experiment_identifier = args.experiment_id
    low_control = args.low_control
    high_control = args.high_control
    id_col = args.id_col
    strain_col = args.strain_col
    
    main(data_converge_path, 
         experiment_identifier,
         low_control,
         high_control,
         id_col=id_col,
         strain_col=strain_col)