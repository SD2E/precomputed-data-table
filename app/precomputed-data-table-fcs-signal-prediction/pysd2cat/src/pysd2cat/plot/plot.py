import pandas as pd
import matplotlib.pylab as plt
from pysd2cat.data import pipeline
import os


def plot_live_dead_control_predictions_by_channel(val_X, val_y, filename='live_dead_control_model_channels.png'):
    """
    This plot is used to analyze the features selected by
    the classifier to make its predictions.
    """
    d = pd.concat([val_X, val_y], axis=1, join='inner')
    #print(d.columns)
    ds = d.sample(frac=0.5)
    c = ds['class_label']
    data = ds[val_X.columns]

    dim = len(val_X.columns)
    fig, axs = plt.subplots(dim, dim, sharex=False, sharey=False, figsize=(50,50))
    plt.subplots_adjust(hspace=1.0)
    for xi, x in enumerate(val_X.columns):
        for yi, y in enumerate(val_X.columns):
            if yi <= xi:
                continue

            plot_predicted_label_scatter(axs[xi, yi], x, y, c, data)
    fig.savefig(filename)
    plt.close()
    #return fig

def plot_predicted_label_scatter(ax, x, y, c, data):
    """
    Used to plot a subplot of two data columns colored by
    the class predictions
    """
    ax.scatter(data[x], data[y], c=c)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    #ax.set_xscale("log", nonposx='clip')
    #ax.set_yscale("log", nonposy='clip')
    #lims = [
    #    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    #]

    #ax.set_xlim(lims)
    #ax.set_ylim(lims)
    #plt.xscale('log')
    #plt.yscale('log')
    #return ax
    #vals = val_X.loc[val_X['BL1-A']]
    #ax.scatter(val_X['BL1-A'], val_X['RL1-A'], c=correct)
    #plt.title("Raw vs. MEFL P(correct)")
    #plt.figure(figsize=(1,1))


    # now plot both limits against eachother
    #ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    #ax.set_aspect('equal')
    #ax.legend()





def plot_live_dead_threshold(plot_df, title, filename='live_dead_control_threshold_accuracy.png'):
    #for od in ods:
    #    if math.isnan(od):
    #        gate_od_df = plot_df.loc[plot_df['od'].isnull()]
    #    else:
    #        gate_od_df = plot_df.loc[plot_df['od'] == od]
    plt.scatter(plot_df['threshold'], plot_df['probability_correct'],label='GFP')
    #plt.fill_between(plot_df['threshold'], gate_od_df['lower_ci'], gate_od_df['upper_ci'],  alpha = 0.4,label=None)
    plt.plot(plot_df['threshold'], plot_df['probability_correct'])

    #plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel('Threshold')
    plt.ylabel('P(Correct)')
    plt.title(title)
    plt.xscale('log')
    plt.savefig(filename)
    plt.close()




    
def plot_predicted_live_scatter(ax, x, y, c, data, title):
    ax.scatter(data[x], data[y], c=c, s=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 10e7)
    ax.set_ylim(1, 10e7)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)

def plot_live_dead(df, title, plot_dir='.'):
    x = ['BL1-A', 'BL1-A'] #GFP
    y = ['RL1-A', 'FSC-A'] #Sytox, FSC
    fig, axs = plt.subplots(len(y), 1, sharex=False, sharey=False, figsize=(10,10))
    fig.tight_layout()
    for ci in range(len(y)):
        plot_predicted_live_scatter(axs[ci], x[ci], y[ci], df['class_label'], df, title)
    fig.savefig(os.path.join(plot_dir,title+".png"))
    plt.close()
    #return fig      


def plot_live_dead_experiment(df, inputs, title, plot_dir='.', pdf='result.pdf'):
    x = ['BL1-A', 'BL1-A'] #GFP
    y = ['RL1-A', 'FSC-A'] #Sytox, FSC
    fig, axs = plt.subplots(len(y), len(inputs), sharex=False, sharey=False, figsize=(10,10))
    fig.tight_layout()
    for ii, input in enumerate(inputs):
        df_i = df.loc[(df['input'] == input)]
        for ci in range(len(y)):
            plot_predicted_live_scatter(axs[ci, ii], x[ci], y[ci], df_i['class_label'], df_i, title+"_"+input)
    #fig.savefig(os.path.join(plot_dir,title+".png"))
    pdf.savefig(fig)
    plt.close()


    
def plot_live_dead_old(circuit, scaler, experiment, model, data_dir=''):
    #circuits = ['NOR', 'OR', 'NAND', 'AND', 'XNOR', 'XOR']
    #circuits = ['NOR', 'OR']
    #inputs = ['NOR-00-Control', '00', '01', '10', '11', 'WT-Dead-Control', 'WT-Live-Control']
    #inputs = ['NOR-00-Control', 'WT-Dead-Control', 'WT-Live-Control']
    inputs = [ '00', '01', '10', '11']
    #inputs = ['00', '01']
    x = ['BL1-A', 'BL1-A'] #GFP
    y = ['RL1-A', 'FSC-A'] #Sytox, FSC
    fig, axs = plt.subplots(len(y), len(inputs), sharex=False, sharey=False, figsize=(20,20))
    #plt.subplots_adjust(hspace=1.0)
    fig.tight_layout()
    fraction=0.05
    for ii, input in enumerate(inputs):
        if input == 'NOR-00-Control' or \
           input == 'WT-Dead-Control' or \
           input == 'WT-Live-Control':
            continue
            title=input
            c_df = pipeline.get_control_dataframe_for_classifier(circuit, input, od=0.0003, media='SC Media', experiment=experiment, data_dir=data_dir)
        else:
            title=circuit+ " " + input + "\nTX " + experiment                
            print(circuit + " " + input)
            c_df = pipeline.get_strain_dataframe_for_classifier(circuit, input, od=0.0003, media='SC Media', experiment=experiment, data_dir=data_dir)
            #c_df = c_df.head()
            #print(c_df)
            #print(c_df.columns)
        c_df_norm = scaler.transform(c_df)
        c_df_y = model.predict(c_df_norm)
        #c_df_y = scaler.inverse_transform(c_df_y)
        #print(c_df.columns)
        #df_X = pd.DataFrame(c_df_norm, columns = c_df.columns)
    
        #print(c_df_y)
        df_y = pd.Series(c_df_y)
        #print(df_y)
        d = pd.concat([c_df, df_y], axis=1, join='inner')
        #print(d)
        
#        from sklearn.cluster import SpectralClustering
#        import numpy as np
#        clustering = SpectralClustering(n_clusters=2,
#                                        assign_labels="discretize",
#                                        random_state=0).fit(c_df)
#        c_df_cl = pd.concat([c_df, clustering.labels_], axis=1, join='inner')
#        print(c_df_cl.head())
        #ds = 
        ds = d.sample(frac=fraction, replace=True)
        c = ds[0]
        data = ds[c_df.columns]

        #print("plotting")
        for ci in range(len(y)):
            plot_predicted_live_scatter(axs[ci, ii], x[ci], y[ci], c, data, title)
    return fig      

    