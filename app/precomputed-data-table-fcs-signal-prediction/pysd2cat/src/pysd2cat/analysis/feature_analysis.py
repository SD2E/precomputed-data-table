import sys
import os
import json
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from pprint import pprint
from pysd2cat.data import pipeline
from matplotlib.ticker import NullFormatter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, SpectralClustering
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from itertools import repeat, islice,cycle
from pysd2cat.analysis.Names import Names
import FlowCytometryTools as FCT

rank = MPI.COMM_WORLD.Get_rank()
####
#Important columns: FSC-H, FSC-W

def axis_formatter(ax,x,y,c):
    ax.scatter(x, y, c=c)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    return ax

#TODO: Consider returning a dictionary
def t_sne(X,perplexity):
    '''
    Compute the first two t-sne components
    :param df: dataframe
    :param perplexity: perplexity value
    :return: tuple of perplexity and transformed T-sne
    '''
    print("Starting t-sne analysis with perplexity: ",perplexity)
    tsne = TSNE(n_components=2, init='random',
                random_state=0, perplexity=perplexity)
    Y = pd.DataFrame(tsne.fit_transform(X),columns=["dim1","dim2"],index=X.index)

    return [perplexity,X.join(Y)]

def write_out_dataframe(results_list):
    labels = ms_cluster(results_list[0][1], results_list[0][1].columns)
    dfs = []
    for result in results_list:
        result[1]["class_label"] = labels
        result[1]["perplexity"] = result[0]
        dfs.append(result[1])
    full_df = pd.concat(dfs)
    print("Writing dataframe")
    full_df.to_csv("Dead_dataframe_with_cluster_labels.csv")

def visualize(results_list,x_colname='FSC-H',y_colname='FSC-W',label_name='class_label'):
    labels = ms_cluster(results_list[0][1], results_list[0][1].columns)


    print(results_list[0][1]["class_label"].value_counts())
    num_figs = len(results_list)
    print("Writing dataframe")
    results_list[0][1].to_csv("Dead_dataframe_with_cluster_labels.csv")
    (fig, subplots) = plt.subplots(1, num_figs+1, figsize=(15, 8), squeeze=False)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(results_list[0][1]['class_label']) + 1))))
    print(int(max(results_list[0][1]['class_label']) + 1))

    for i,result in enumerate(results_list):
        if i==0:
            ax = subplots[0][0]
            ax.scatter(result[1][x_colname],result[1][y_colname],s=10,color=colors)
            ax.set_title("Flow data with channels " + x_colname + " x " + y_colname)

        ax = subplots[0][i+1]
        ax.set_title("Perplexity=%d" % result[0])
        ax.scatter(result[1]["dim1"],result[1]["dim2"],s=10,color=colors[list(results_list[0][1]["class_label"])])

    print("Saving figure...")
    plt.savefig("Tsne_plot_dead.png")
    plt.close()



def ms_cluster(df, cols, bandwidth=None):
    X = df[cols].values
    if bandwidth == None:
        bandwidth = estimate_bandwidth(X)
    # setting bin_seeding=False takes way too long for large datasets
    #X = StandardScaler().fit_transform(X)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    return labels

def analyze(df):
    '''
    Run a clustering and tsne analysis with multiple perplexities to see what the output data looks like
    :param df: dataframe to perform T-SNE on
    :param label_name: the name of the column that has the label
    :return: nothing, just a T-SNE plot
    '''

    #Get the numeric columns and get ready for clustering


    perplexities = [10, 20]
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            print("Running t-sne")
            results = executor.map(t_sne, repeat(df), perplexities)
    if rank == 0:
        results_list = list(results)
        return results_list
    else:
        return 0




def main():
    ## Where data files live
    ##HPC
    data_dir = '/work/projects/SD2E-Community/prod/data/uploads/'
    ##Jupyter Hub
    # data_dir = '/home/jupyter/sd2e-community/'

    print("Building Live/Dead Control Dataframe...")
    #live_dead_df = pipeline.get_flow_dataframe(data_dir,filename="WT-Dead-Control__.fcs")
    live_dead_df = pipeline.get_dataframe_for_live_dead_classifier(data_dir,control_type=[Names.WT_DEAD_CONTROL],fraction=.01,max_records=1000)
    live_dead_df = live_dead_df.drop(columns=['class_label'])
    nrows = len(live_dead_df)
    ncols = len(live_dead_df.columns)
    print("Dataframe constructed with {0} rows and {1} columns".format(nrows,ncols))
    live_dead_df = live_dead_df.head(n=3000)
    results = analyze(live_dead_df)
    if rank ==0:
        write_out_dataframe(results)


if __name__ == '__main__':
    main()


    ###TESTING WITH SCI-KIT DATA####
    #X, y = datasets.make_circles(n_samples=300, factor=.5, noise=.05)
    #df = pd.DataFrame(np.column_stack((X,y)),columns=['col_'+ str(i) for i in range(0,X.shape[1]+1)])
    #df = df.rename(columns={'col_'+str(X.shape[1]):'class_label'})
    #tsne_analysis(df,x_colname='col_0',y_colname='col_1')
