
# Run clustering at scale for 100s of thousands of cells in 12D space

import pandas as pd
import random
import openensembles as oe
import sys
import os
import json
import math
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from pprint import pprint
#from pysd2cat.data import pipeline
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

def Jaccard_total(set1, set2):
    inter = float(len(set1.intersection(set2)))
    return inter / (len(set1) + len(set2) - inter)

def compare_two_labelings(df, entity_col, clust_col_1, clust_col_2):
    clust1 = np.unique(df[clust_col_1])
    clust2 = np.unique(df[clust_col_2])
    clust1_list = []
    clust2_list = []
    jaccard_list = []
    for x in clust1:
        clust1_list.append(get_entities_with_label(df, entity_col, clust_col_1, x))
    for y in clust2:
        clust2_list.append(get_entities_with_label(df, entity_col, clust_col_2, y))
    if len(clust1) >= len(clust2):
        for a, z in zip(clust1_list, range(len(clust1_list))):
            jaccard_list.append([])
            for b in clust2_list:
                jaccard_list[z].append(analysis.Jaccard_total(set(a), set(b)))
    else:
        for a, z in zip(clust2_list, range(len(clust2_list))):
            jaccard_list.append([])
            for b in clust1_list:
                jaccard_list[z].append(analysis.Jaccard_total(set(a), set(b)))
    return jaccard_list

def ms_cluster(df, cols, bandwidth=None):
    X = df[cols].values
    if bandwidth == None:
        bandwidth = estimate_bandwidth(X)
    # setting bin_seeding=False takes way too long for large datasets
    X = StandardScaler().fit_transform(X)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    #dataframe of points --> cluster label
    df['c_label'] = ms.labels_
    return df


##TODO: Finish Ensemble clustering
#1. Create large dataframe -- DONE
#2. Randomly sample points and cluster -- DONE
#2a. Create dictionary of cluster label as key and value as set of point indexes
#2b. Output the dataframe of points by cluster label -- done
#3. Compute adjacency matrix of cluster label from each clustering
#4. Run hierarchical clustering on that adjacency matrix and generate metacluster_index--cluster_index map
#5. Create dataframe of point -- meta_cluster index
#6. Vote on each meta_cluster index to assign each point to the max meta cluster index
#7. Remove points with no meta_cluster index
#8.


def main():
    data_dir = '/work/projects/SD2E-Community/prod/data/uploads/'

    ######################
    ## TEST DATA
    ######################
    n_samples = 400
    X, y = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=0.02, random_state=None)
    df = pd.DataFrame(X)
    df_sub = df.sample(frac=0.1)

    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is not None:
            print("Running clustering")
            cluster_results = list(executor.map(ms_cluster, df_sub, df_sub.columns))
            df_all = pd.concat(cluster_results,axis=1)##Need to adjust the names here and don't need to join the entire dataframe
            print("Total down select of overall dataframe is:")
            print(len(df_sub)/len(df_all))














if __name__ == '__main__':
    main()