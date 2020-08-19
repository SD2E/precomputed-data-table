import math
import matplotlib.pyplot as plt                   # For graphics
import numpy as np
import seaborn as sns
import pandas as pd  


volume_to_per =  {
    'experiment.uw_biofab.34600' : {
            0 : 0,
            29.: .029,
            64 : .064,
            105: .105,
            170: .170,
            250: .25,
            370: .370,
            570: .570,
            980: .980
            },
    'experiment.uw_biofab.34601' :{
            0 : 0,
            29: .029,
            64 : .064,
            105: .105,
            170: .170,
            250: .25,
            370: .370,
            570: .570,
            980: .980
            },
    'r1dd37mcxv5pf4' : {
            0 : 0,
            29: 29/279,
            64 : 64/314,
            105: 105/355,
            170: 170/420,
            250: 250/500,
            370: 370/620,
            570: 570/820,
            980: 980/1230
            },
    'r1dk8xp9dymm54' : {
            0 : 0,
            300 : 300/2000
            },
    'r1dmsrursbqwuz' : { x: x/(1400.0) for x in [0.0, 140.0, 210.0, 280.0, 1120.0]}
}

def get_statistics_by_volume(leader_board_df, experiment_strain=None, experiment_lab=None):
    metrics=['Balanced Accuracy', 'F1 Score']
    stains=['SYTOX Red Stain', None]
    fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=(4*len(metrics)+4, 4), dpi=200)
    experiments = leader_board_df.experiment_id.dropna().unique()
    print(experiments)
    plot_df=leader_board_df#leader_board_df.loc[leader_board_df.stain == True]

    for j, col in enumerate(ax):
    
        #for rs in plot_df.kill.unique():
        df = plot_df


        #xvals=df['kill']


        #col.set_xlabel("Ethanol Volume (uL)")
        col.set_xlabel("Ethanol %")
        col.set_ylabel(metrics[j])

        for experiment in experiments:
            for stain in stains:
                #print(experiment + " " + str(stain))
                if experiment_strain is not None and experiment_lab is not None:
                    label = experiment_lab[experiment] + ", " + experiment_strain[experiment] + ", " + str(stain)
                else:
                    label = experiment + " " + str(stain)
                if stain is None:
                    mdf=df.loc[(df['experiment_id'] == experiment) & (df['stain'].isna())]
                else:
                    mdf=df.loc[(df['experiment_id'] == experiment) & (df['stain'] == stain)]                    
                mdf=mdf.drop_duplicates()
                #print(mdf.head(1))

                xvals=mdf['dead_volume'].apply(lambda x: volume_to_per[experiment][x])
                xvals1=mdf.groupby(['dead_volume']).agg(np.mean).reset_index()['dead_volume'].apply(lambda x: volume_to_per[experiment][x])
                yvals=mdf[metrics[j]]
                yvals1=mdf.groupby(['dead_volume']).agg(np.mean)[metrics[j]]
                #plt.xtics(xvals)

                #col.scatter(xvals, yvals,s=100, alpha=0.5, label=label)
                col.scatter(xvals, yvals, label=label, alpha = 0.5)
                col.plot(xvals1, yvals1, alpha = 0.5)
        lims = [ .65, 1
        ]
        
        #col.plot(lims, lims, 'k-', alpha=0.1, zorder=0)
        col.set_title("Model(0.0 = Live, X% = Dead)\n Five Train/Test Splits")
        #col.set_xscale('log')
        #col.set_xlim([0.65, 1])
        #col.set_ylim([0.65, 1])
        #col.set(adjustable='box', aspect='equal')
        #plt.axis([0.85, 1, 0.85, 1])
        plt.legend(bbox_to_anchor=(1.0, 1.0),
          ncol=1)

        #plt.axis('equal')
        #plt.gca().set_aspect('equal', adjustable='box')



    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return fig

def get_statistics_by_time(leader_board_df, experiment_strain=None, experiment_lab=None):
    metrics=['Balanced Accuracy', 'F1 Score']
    stains=['SYTOX Red Stain', None]
    fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=(4*len(metrics)+4, 4), dpi=200)
    experiments = leader_board_df.experiment_id.dropna().unique()
    dead_volumes = leader_board_df.dead_volume.dropna().unique()
    dead_volumes.sort()
 #   print(experiments)
    plot_df=leader_board_df#leader_board_df.loc[leader_board_df.stain == True]

    for j, col in enumerate(ax):
    
        #for rs in plot_df.kill.unique():
        df = plot_df


        #xvals=df['kill']


        #col.set_xlabel("Ethanol Volume (uL)")
        col.set_xlabel("Time (h)")
        col.set_ylabel(metrics[j])

        for experiment in experiments:
            for stain in stains:
                for dead_volume in dead_volumes:
                    #print(experiment + " " + str(stain))
                    if experiment_strain is not None and experiment_lab is not None and dead_volume is not None:
                        label = experiment_lab[experiment] + ", " + experiment_strain[experiment] + ", " + str(stain) + ", " + "{:.2f}".format(volume_to_per[experiment][dead_volume])
                    else:
                        label = experiment + " " + str(stain)
                    if stain is None:
                        mdf=df.loc[(df['experiment_id'] == experiment) & (df['stain'].isna()) & (df['dead_volume'] == dead_volume)]
                    else:
                        mdf=df.loc[(df['experiment_id'] == experiment) & (df['stain'] == stain) & (df['dead_volume'] == dead_volume)]                    
                    mdf=mdf.drop_duplicates().sort_values(by=['time_point'])
                    #print(mdf.head(1))

                    xvals=mdf['time']
                    xvals1=mdf.groupby(['time']).agg(np.mean).reset_index()['time']
                    yvals=mdf[metrics[j]]
                    yvals1=mdf.groupby(['time']).agg(np.mean)[metrics[j]]
                    #plt.xtics(xvals)

                    #col.scatter(xvals, yvals,s=100, alpha=0.5, label=label)
                    col.scatter(xvals, yvals, label=label, alpha = 0.5)
                    col.plot(xvals1, yvals1, alpha = 0.5)
        lims = [ 0, 1
        ]
        
        #col.plot(lims, lims, 'k-', alpha=0.1, zorder=0)
        col.set_title("Time series Model(0.0 = Live, X% = Dead)\n Five Train/Test Splits")
        #col.set_xscale('log')
        #col.set_xlim([0.65, 1])
        #col.set_ylim([0.65, 1])
        #col.set(adjustable='box', aspect='equal')
        #plt.axis([0.85, 1, 0.85, 1])
        plt.legend(bbox_to_anchor=(1.0, 1.0),
          ncol=1)

        #plt.axis('equal')
        #plt.gca().set_aspect('equal', adjustable='box')



    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return fig


def get_channel_mean_titration(experiment_df,
                               only_live=False,
                               channels=['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A',
                                         'FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'FL4-H'],
                               lab='BioFab',
                               stat=np.mean
                               ):
    # Overlay the channels at different concentrations to see how they shift

    stains=experiment_df.stain.unique()
    print(stains)
    
    channels.sort()

    #fig = plt.figure( dpi=200)

    fig, ax = plt.subplots(nrows=1, ncols=len(stains), figsize=(4*len(stains)+4, 4), dpi=200)
    experiments = experiment_df.experiment_id.dropna().unique()
    if 'strain' in experiment_df.columns:
        experiment_id = experiment_df.strain.dropna().unique()[0]
    else:
        experiment_id = experiment_df.experiment_id.dropna().unique()[0]        
    for j, col in enumerate(ax):
        if only_live:
            df = experiment_df.loc[experiment_df['live'] == 1]
        else:
            df = experiment_df

        if type(stains[j]) is not str and math.isnan(stains[j]):
            plot_df = df.loc[df['stain'].isna()].groupby(['kill_volume']).agg(stat).reset_index()
            col.set_title("Mean Intensity " + str(experiment_id) + ", Stain: None")

        else:
            plot_df = df.loc[df['stain'] == stains[j]].groupby(['kill_volume']).agg(stat).reset_index()
            col.set_title("Mean Intensity " + str(experiment_id) + ", Stain: " + str(stains[j]))
        col.set_xlabel("Ethanol %")
        col.set_ylabel("Mean Intensity")


        print(df.head(1))
        for j, channel in enumerate(channels):
            if lab == 'BioFab':
                col.plot(plot_df['kill_volume'].apply(lambda x: x/(1000.0)),
                        plot_df[channel], label=channel)
            elif lab == 'TX':
                col.plot(plot_df['kill_volume'].apply(lambda x: x/(x+250.0)),
                        df[channel], label=channel)
            else:
                raise("Don't know how to calculate % for lab: "+ lab)

        col.set_yscale('log')
        #ax.set_xscale('log')
    plt.legend( bbox_to_anchor=(1.0, 1.0),
              ncol=1)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return fig

def get_channel_mean_timeseries(experiment_df,
                               channels=['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A',
                                         'FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'FL4-H'],
                               lab='BioFab',
                               stat=np.mean
                               ):
    # Overlay the channels at different concentrations to see how they shift

    stains=experiment_df['stain', ''].unique()
    print(stains)

    volumes=experiment_df['kill_volume', ''].unique()
    channels.sort()

    #fig = plt.figure( dpi=200)

    fig, ax = plt.subplots(ncols=len(volumes), nrows=len(stains), figsize=(4*len(volumes)+4, 4*len(stains)+4), dpi=100)
#    experiments = experiment_df.experiment_id.dropna().unique()
#    if 'strain' in experiment_df.columns:
#        experiment_id = experiment_df.strain.dropna().unique()[0]
#    else:
#        experiment_id = experiment_df.experiment_id.dropna().unique()[0]        
    for j, col in enumerate(ax):
        for i, row in enumerate(col):

            row.set_xlabel("Time (h)")
            row.set_ylabel("Mean Intensity")

            plot_df = experiment_df.loc[(experiment_df['stain', ''] == stains[j]) & (experiment_df['kill_volume', ''] == volumes[i])].sort_values(by=['time'])
            
            for c, channel in enumerate(channels):
                row.plot(plot_df['time'],
                            plot_df[channel, 'mean'], label=channel)
                row.fill_between(plot_df['time'], plot_df[channel, 'mean']-plot_df[channel, 'std'], plot_df[channel, 'mean']+plot_df[channel, 'std'], alpha=0.05)

            #row.set_yscale('log')
            row.set_title("Stain: {} Ethanol Vol: {}".format(stains[j], volumes[i]))
            #row.set_xlim([0., 1])
            #row.set_ylim([1e-2, 1e6])
            row.set_ylim([0, 15])

            #ax.set_xscale('log')
    plt.legend( bbox_to_anchor=(1.0, 1.0),
              ncol=1)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return fig

def get_stain_scatter(leader_board_df, experiment_id, experiment_strain=None):
    metrics=['Balanced Accuracy', 'F1 Score']

    fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=(4*len(metrics), 4), dpi=200)
    #plt.rcParams['font.size'] = 1


    leader_board=leader_board_df.loc[leader_board_df['experiment_id'] == experiment_id]

    stain = leader_board.loc[(leader_board['stain'] == "SYTOX Red Stain") ]
    no_stain = leader_board.loc[(leader_board['stain'].isna()) ]

    plot_df = stain.merge(no_stain, on=['experiment_id', 'random_state', 'dead_volume'], how='inner')

    for j, col in enumerate(ax):
        col.set_xlabel("With Stain")
        col.set_ylabel("Without Stain")

        for rs in plot_df.random_state.unique():
            df = plot_df.loc[plot_df['random_state'] == rs]

            xvals=df[metrics[j]+"_x"]
            yvals=df[metrics[j]+"_y"]


            col.scatter(xvals, yvals,s=100, alpha=0.5, label=rs)

        lims = [ 0, 1]

        col.plot(lims, lims, 'k-', alpha=0.1, zorder=0)
        col.set_title(experiment_strain[experiment_id])
#        col.set_xlim([0., 1])
#        col.set_ylim([0., 1])
#        col.set(adjustable='box', aspect='equal')
        #plt.axis([0.85, 1, 0.85, 1])
    plt.legend()
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    stats = plot_df.apply(lambda x: x['Balanced Accuracy_x'] - x['Balanced Accuracy_y'], axis=1).agg([np.mean, np.std])


    
    return fig, stats


def get_channel_histograms(experiment_df, stain='SYTOX Red Stain',
                               channels=['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A', 'FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'FL4-H']
):
    # Get all channels from a sample and plot as histogram.

    
    volumes=experiment_df.kill_volume.dropna().unique()
    volumes.sort()
    channels.sort()

    fig = plt.figure(figsize=(4*len(volumes), 4*len(channels)), dpi=200)

    bins = [10**x for x in range(0, 10) ] 
    bins.sort()
    live_output_col='live'
    #for volume in volumes:
    for i, volume in enumerate(volumes):
        if stain is None:
            df = experiment_df.loc[(experiment_df['dead_volume']==volume) & (experiment_df['stain'].isna())]
        else:
            df = experiment_df.loc[(experiment_df['kill_volume']==volume)& (experiment_df['stain'] == stain)]            

        for j, channel in enumerate(channels):
            ax = fig.add_subplot(len(channels), len(volumes), (j*len(volumes))+i+1)



            #volume = volumes[j]
            #for rs in plot_df.kill.unique():

            #xvals=df[channel]
            #xvals=df['kill']


            ax.set_xlabel(channel + " Intensity")
            ax.set_ylabel("Frequency")

            #col.scatter(xvals, yvals,s=100, alpha=0.5)
            ax.hist(df[channel], bins=100, alpha=0.5, label="All")
            ax.hist(df.loc[df[live_output_col]==1][channel], bins=100, alpha=0.5, label="Live")
            lims = [ .65, 1
            ]

            #col.plot(lims, lims, 'k-', alpha=0.1, zorder=0)
            ax.set_title(channel + " Intensity, Ethanol " + str(volume) + "uL")
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlim([1, 10e9])
            ax.set_ylim([1, 10e6])
            #col.set(adjustable='box', aspect='equal')
            #plt.axis([0.85, 1, 0.85, 1])
            plt.legend()
            #plt.axis('equal')
            #plt.gca().set_aspect('equal', adjustable='box')

    #plt.axis([0, 10e7, 0, 10e6])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)





def get_timeseries_multi_scatter(xcol, ycol, stain, time_points, kill_volumes, prediction_files, flow_data, live_col='live', frac=0.1, kdeplot=False):
    ## FIXME Assumes that kill_volumes are the classes

    colors = ["Blue", "Green", "Yellow", "Orange", "Red"]
    
    ## track number of points classified as class i for each volume
    num_points = { i : {k: [] for k in kill_volumes} for i, _ in enumerate(kill_volumes)}

    fig, ax = plt.subplots(ncols=len(time_points), nrows=len(kill_volumes), figsize=(4*len(time_points), 4*len(kill_volumes)), dpi=200)
    for i, row in enumerate(ax):
        kill_volume = kill_volumes[i]
        #print(kill_volume)
        for j, col in enumerate(row):

            time_point = time_points[j]
            #print(time_point)
            time = flow_data.loc[flow_data.time_point == time_point].time.iloc[0]
            df = pd.read_csv(prediction_files[time_point], index_col=0)


            plot_df = df.loc[(df.stain == stain) & (df.kill_volume == kill_volume)]

            for c, _ in enumerate(kill_volumes):
                live_df = plot_df.loc[plot_df[live_col] == c]
                num_points[c][kill_volume].append(len(live_df))

                live_df = live_df.sample(frac=frac)
                live_df = live_df[[xcol, ycol]].apply(np.log).replace([-np.inf, np.inf], np.nan).dropna(how="any")
                try:
                    if kdeplot:
                        sns.kdeplot(live_df[xcol], live_df[ycol], ax=col, alpha=0.5, cmap="Blues", shade=True, label=str(kill_volumes[c]), shade_lowest=False, 
                        dropna=True)
                    else:
                        col.scatter(live_df[xcol], live_df[ycol], c=colors[c], label=str(kill_volumes[c]), s=1, alpha=0.1)
                    col.legend()
                except Exception as e:
                    pass
            col.set_xlabel("log({})".format(xcol))
            col.set_ylabel("log({})".format(ycol))
        #        col.set_xscale('log')
        #        col.set_yscale('log')
            col.set_xlim(0, 15)
            col.set_ylim(0, 15)
            col.set_title("Kill Volume (uL): " + str(kill_volume) + "Time (h): " + str(time))
            #break


    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return num_points


def get_timeseries_scatter(xcol, ycol, stain, time_points, kill_volumes, prediction_files, flow_data, live_col='live', frac=0.1, kdeplot=False):
    num_points = {'dead' : {k: [] for k in kill_volumes}, "live" : {k: [] for k in kill_volumes}}
    fig, ax = plt.subplots(ncols=len(time_points), nrows=len(kill_volumes), figsize=(4*len(time_points), 4*len(kill_volumes)), dpi=200)
    for i, row in enumerate(ax):
        kill_volume = kill_volumes[i]
        #print(kill_volume)
        for j, col in enumerate(row):

            time_point = time_points[j]
            #print(time_point)
            time = flow_data.loc[flow_data.time_point == time_point].time.iloc[0]
            df = pd.read_csv(prediction_files[time_point], index_col=0)


            plot_df = df.loc[(df.stain == stain) & (df.kill_volume == kill_volume)]
            live_df = plot_df.loc[plot_df[live_col] == 1]
            dead_df = plot_df.loc[plot_df[live_col] == 0]
            #live_df = plot_df.loc[plot_df.live == i]

            num_points['dead'][kill_volume].append(len(dead_df))
            num_points['live'][kill_volume].append(len(live_df))

            live_df = live_df.sample(frac=frac)
            live_df = live_df[[xcol, ycol]].apply(np.log).replace([-np.inf, np.inf], np.nan).dropna(how="any")
            dead_df = dead_df.sample(frac=frac)
            dead_df = dead_df[[xcol, ycol]].apply(np.log).replace([-np.inf, np.inf], np.nan).dropna(how="any")


            #print(live_df[xcol].unique())
            #print(live_df[ycol].unique())
            #print(dead_df[xcol].unique())
            #print(dead_df[ycol].unique())

            try:
                if kdeplot:
                    sns.kdeplot(live_df[xcol], live_df[ycol], ax=col, alpha=0.5, cmap="Blues", shade=True, label="Live", shade_lowest=False, 
                        dropna=True)
                else:
                    col.scatter(live_df[xcol], live_df[ycol], c="Blue", label="Live", s=1, alpha=0.1)
            except Exception as e:
                pass
            try:
                if kdeplot:
                    sns.kdeplot(dead_df[xcol], dead_df[ycol], ax=col,  alpha=0.5, cmap="Oranges", shade=True, label="Dead",shade_lowest=False,
                        dropna=True)
                else:
                    col.scatter(dead_df[xcol], dead_df[ycol], c ="Red", label="Dead", s=1, alpha=0.1)
                col.legend()
            except Exception as e:
                pass
            col.set_xlabel("log({})".format(xcol))
            col.set_ylabel("log({})".format(ycol))
        #        col.set_xscale('log')
        #        col.set_yscale('log')
            col.set_xlim(0, 15)
            col.set_ylim(0, 15)
            col.set_title("Kill Volume (uL): " + str(kill_volume) + "Time (h): " + str(time))
            #break


    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return num_points
