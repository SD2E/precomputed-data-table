import pandas as pd
from sklearn.manifold import TSNE
from pprint import pprint
import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from itertools import repeat, islice,cycle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True, help='input file')
parser.add_argument('--output_file',required=True, help='input file')
parser.add_argument('--delim',required=False,default=' ',help='delimiter used for the input file')
parser.add_argument('--header',required=False,help='header in file, pass string None if no header')
parser.add_argument('--index',required=False,help='column to use as index')

args = parser.parse_args()

rank = MPI.COMM_WORLD.Get_rank()


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
    dfs = []
    for result in results_list:
        result[1]["perplexity"] = result[0]
        dfs.append(result[1])
    full_df = pd.concat(dfs)
    print("Writing dataframe")
    output_file = args.output_file
    full_df.to_csv(output_file)


def analyze(df):
    '''
    Run a clustering and tsne analysis with multiple perplexities to see what the output data looks like
    :param df: dataframe to perform T-SNE on
    :param label_name: the name of the column that has the label
    :return: nothing, just a T-SNE plot
    '''

    #Get the numeric columns and get ready for clustering


    perplexities = [5,10, 20,30]
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
    fname = args.input_file
    if '.csv' in fname:
        delim = ','
    elif '.tsv' in fname:
        delim = '\t'
    else:
        delim = args.delim
    header=args.header
    index = args.index
    if header == 'None':
        df = pd.read_csv(fname,delimiter=delim,header=None,skiprows=1)
        df.columns = [str(col) for col in df.columns]
    else:
        df = pd.read_csv(fname, delimiter=delim)
    if index:
        df.set_index([index],inplace=True)
    nrows = len(df)
    ncols = len(df.columns)
    print("Dataframe constructed with {0} rows and {1} columns".format(nrows,ncols))
    results = analyze(df)
    if rank ==0:
        write_out_dataframe(results)


if __name__ == '__main__':
    ##To run this use the following command on an HPC cluster:
    #Make sure you are in the project folder
    #ibrun -n <NUM_TASKS> python3 src/pysd2cat/analysis/tsne_analysis.py --input_file <PATH_TO_FILE>
    main()


    ###TESTING WITH SCI-KIT DATA####
    #X, y = datasets.make_circles(n_samples=300, factor=.5, noise=.05)
    #df = pd.DataFrame(np.column_stack((X,y)),columns=['col_'+ str(i) for i in range(0,X.shape[1]+1)])
    #df = df.rename(columns={'col_'+str(X.shape[1]):'class_label'})
    #tsne_analysis(df,x_colname='col_0',y_colname='col_1')
