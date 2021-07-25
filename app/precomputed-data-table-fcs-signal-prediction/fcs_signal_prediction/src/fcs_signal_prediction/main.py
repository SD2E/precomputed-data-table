import argparse
import numpy as np
from typing import Optional
from fcs_signal_prediction.predict import predict_signal
from fcs_signal_prediction.utils import data_utils as du

parser = argparse.ArgumentParser(description='Predict Signal Output for samples')
parser.add_argument('data_converge_path', type=str,
                    help='Path to data converge results')
parser.add_argument('experiment_id', type=str,
                    help='Experiment Identifier')
parser.add_argument('low_control', type=str,
                    help='Strain Name of Low Control')
parser.add_argument('high_control', type=str,
                    help='Strain Name of High Control')
parser.add_argument('hl_idx', type=int,
                    help='The High/Low control combo index')
parser.add_argument('--id_col', type=str,
                    help='Sample id column name', default="sample_id")
parser.add_argument('--strain_col', type=str,
                    help='Strain column name', default="strain_name")





def main(data_converge_path: str, 
         experiment_identifier: str,
         low_control : str,
         high_control : str,
         hl_idx: str,
         id_col : Optional[str]="sample_id",
         strain_col : Optional[str]='strain_name'):
    """
    Get the raw flow cytometry data, predict the output signal for each event,
    aggregate by sample, return metadata with columns for mean and std dev. of
    the predicted signal for each sample.
    """
    
    meta = du.get_meta(data_converge_path, du.get_record(data_converge_path))
    plate_samples_dict = du.make_plate_samples_dict(meta)
    plate_ids_in_record = du.get_plate_ids(data_converge_path, experiment_identifier)
    ## check record ids match ids in meta and print confirmation along with number of samples per plate
    du.match_exp_ids(plate_samples_dict, plate_ids_in_record)
    
    import pandas as pd
    import os, json
    ## Run FSP on each plate separately.
    
    results_fname_list = list()

    for plate_id in plate_ids_in_record:

        ## Get the data
        df = du.grab_plate_samples_df(plate_id, plate_samples_dict, data_converge_path)
        print(f'Total events collected from plate {plate_id}: {df.shape[0]}')
        meta_data_df = df.merge(meta[[id_col, strain_col]])
        
    
        ## Get the channels in the data
        
        channels = list(meta_data_df.columns)
        channels.remove(id_col)
        channels.remove(strain_col)
        
        ## Predict the output signal for each event
        print(f'\tPredicting on {plate_id}')
        pred = predict_signal(meta_data_df, 
                            experiment_identifier,
                            low_control,
                            high_control,
                            id_col,
                            channels,
                            strain_col=strain_col)

        print('\tlog10 normalizing and binning data')
        pred_data_df = pd.concat([df, pred.drop(labels='sample_id', axis=1)], axis=1)
        pred_data_df.to_csv('pred_data.csv')
        print('\tCreating "pON" histograms')
        on_df = du.make_log10_df(pred_data_df, 1)
        print('\tCreating "pOFF" histograms')
        off_df = du.make_log10_df(pred_data_df, 0)
        print('\tCombining "pON" and "pOFF" histogram data')
        log10_pred_df = pd.concat([on_df, off_df])
        log10_pred_df.sort_values(by=['sample_id'], inplace=True)

        log10_pred_fname = 'pdt_{}__{}_HL{}_fcs_signal_prediction__fc_raw_log10_stats.csv'.format(experiment_identifier, plate_id.split('.')[-1], hl_idx+1)
        log10_pred_df.to_csv(log10_pred_fname, index=False)
        results_fname_list.append(log10_pred_fname)

        ## Get the mean and std output signal for each sample
        mean_prediction = pred.groupby([id_col]).agg({"predicted_output" : [np.mean, np.std]}).reset_index()
        mean_prediction.columns  = mean_prediction.columns.map('_'.join)
        mean_prediction = mean_prediction.rename(columns={id_col+"_": id_col})
        

        ## Get counts of each output signal prediction for each sample
        pred['predicted_output'] = pred['predicted_output'].astype(str)
        pred = pred.replace(['0', '1'], ['OFF', 'ON'])
        pred = pred.groupby(['sample_id', 'predicted_output'])['predicted_output'].agg('count').reset_index(name="p")
        pred = pred.pivot(index='sample_id', columns='predicted_output').reset_index()
        pred.columns = pred.columns.map(''.join)

        ## Attach the mean and std to the metadata
        
        meta_pred_stats = meta.merge(mean_prediction, on=id_col)
        result = meta_pred_stats.merge(pred, on='sample_id')

        # well_timeseries = result.groupby(['timepoint', 'well', 'experiment_id']).agg(np.mean).sort_values(by=['well', 'timepoint', 'experiment_id']).reset_index()
        # well_timeseries_fig = plot.plot_well_timeseries(well_timeseries)
        #
        # samples_and_controls_fig = plot.plot_samples_and_controls(df[[id_col, 'BL1-A']].merge(meta[[strain_col, id_col, "well", "timepoint", 'experiment_id']], on=id_col), result, low_control, high_control)

        result['high_control'] = high_control
        result['low_control'] = low_control
        results_fname = 'pdt_{}__{}_HL{}_fcs_signal_prediction.csv'.format(experiment_identifier, plate_id.split('.')[-1], hl_idx+1)

        result.to_csv(results_fname, index=False)

        results_fname_list.append(results_fname)

    return results_fname_list
        # return result, well_timeseries_fig, samples_and_controls_fig



if __name__ == '__main__':
    args = parser.parse_args()

    data_converge_path = args.data_converge_path
    experiment_identifier = args.experiment_id
    low_control = args.low_control
    high_control = args.high_control
    hl_idx = args.hl_idx
    id_col = args.id_col
    strain_col = args.strain_col
    
    main(data_converge_path, 
         experiment_identifier,
         low_control,
         high_control,
         hl_idx,
         id_col=id_col,
         strain_col=strain_col)