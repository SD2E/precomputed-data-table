import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_log_gfp(df, channel, controls):
    df['log ' + channel] = df[channel].apply(lambda x: x+1.0).apply(np.log10)
    if controls.empty:
        return df.replace([np.inf, -np.inf], np.nan).dropna()
    
    controls['log ' + channel] = controls[channel].apply(lambda x: x+1.0).apply(np.log10)
    return (df.replace([np.inf, -np.inf], np.nan).dropna(), controls.replace([np.inf, -np.inf], np.nan).dropna())

def plot_samples_and_controls(df, result, low_control, high_control, channel, controls=pd.DataFrame(), num=10000):
    wells = df.well_id.unique()
    wells.sort()
    num_wells = len(wells)

    plates = df['experiment_id'].unique()
    num_plates = len(plates)
    
    timepoints = df.timepoint.unique()
    timepoints.sort()
    
    fig, ax = plt.subplots(num_wells*num_plates, 1, figsize=(4, 5*num_wells))

    if controls.empty:
        log_df = get_log_gfp(df, channel, controls)
        high_df = log_df.loc[log_df.strain_name == high_control].sample(n=num)
        low_df = log_df.loc[log_df.strain_name == low_control].sample(n=num)
    else:
        high_control = "WT-Live-Control"
        low_control = "WT-Dead-Control"
        (log_df, log_all_controls) = get_log_gfp(df, channel, controls)
        high_df = log_all_controls.loc[log_all_controls.strain_name == high_control].sample(n=num)
        low_df = log_all_controls.loc[log_all_controls.strain_name == low_control].sample(n=num)

    for j, plate in enumerate(plates):
        for i, well in enumerate(wells):
            ax[i+(j*num_wells)].hist(high_df['log ' + channel], label="high", histtype="step")
            ax[i+(j*num_wells)].hist(low_df['log ' + channel], label="low", histtype="step")


            for timepoint in timepoints:
                sample_df = log_df.loc[(log_df.well_id == well) & (log_df.timepoint == timepoint) &(log_df['experiment_id'] == plate)]
                if len(sample_df) > 0:
                    sample_df = sample_df.sample(n=min(num, len(sample_df)))
                    ax[i+(j*num_wells)].hist(sample_df['log ' + channel], label="@{}".format(timepoint), alpha=0.25)


            ax[i+(j*num_wells)].set_xlim(0, 8)
            ax[i+(j*num_wells)].text(9, 0, "\n".join(result[(result.well_id==well)&(result['experiment_id']==plate)][["timepoint", "predicted_output_mean", "predicted_output_std"]].transpose()[0:].to_string().split("\n")[1:]))
            ax[i+(j*num_wells)].set_title("{} {}".format(plate, well))
            plt.legend()
    plt.subplots_adjust(hspace=0.5)
    
    return fig



def plot_well_timeseries(well_timeseries):
    wells = well_timeseries.well_id.unique()
    num_wells = len(wells)

    plates = well_timeseries['experiment_id'].unique()
    num_plates = len(plates)

    
    fig, ax = plt.subplots(num_wells*num_plates, 1, figsize=(3, 3*num_wells*num_plates))

    for j, plate in enumerate(plates):
        for i, well in enumerate(wells):
            #print(i)
            sample_df = well_timeseries.loc[(well_timeseries.well_id == well) & (well_timeseries['experiment_id'] == plate)]

            #ax[i].plot(sample_df["timepoint"], sample_df["predicted_output_mean"])
            ax[i+(j*num_wells)].errorbar(sample_df["timepoint"], sample_df["predicted_output_mean"], yerr=sample_df["predicted_output_std"])
            ax[i+(j*num_wells)].set_ylim(0, 1.0)
            ax[i+(j*num_wells)].set_title("{} {}".format(plate, well))
    return fig