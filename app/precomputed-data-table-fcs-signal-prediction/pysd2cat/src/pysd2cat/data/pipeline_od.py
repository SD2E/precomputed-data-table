from pysd2cat.data import pipeline
import pandas as pd
import os
import glob
from os.path import expanduser

############################################################################
## ETL related data functions available from pipeline
############################################################################


def get_etl_od_data():
    """
    Return dictionary mapping experiment id to data file
    """
    data_loc = 'sd2e-community/shared-q1-workshop/gzynda/platereactor_out/transcriptic/201808/yeast_gates'
    experiments = glob.glob(os.path.join(expanduser("~"), data_loc, '*'))
    results = {}
    for experiment in experiments:
        files = glob.glob(os.path.join(experiment, 'pyPlateCalibrate*', '*scaled_data.csv'))        
        ex_id = 'experiment.transcriptic.' + experiment.split('/')[-1]
        results[ex_id] = files
    return results

def get_meta_and_etl_od_data(experiment_id, od_files):
    """
    Create joint dataframe with metadata and etl'd od data
    """
    samples = pipeline.get_experiment_samples(experiment_id,file_type='CSV')
    calibration_samples = [x for x in samples if 'calibration' in x['filename']]
    ## Drop calibration_samples for now
    samples = [x for x in samples if x not in calibration_samples]
    
    assert(len(samples) > 0)
    
    #print(calibration_samples)
    #print("Got " + str(len(samples)) + " samples for " + experiment_id)

    meta = pipeline.get_metadata_dataframe(samples)
    #od_files = meta.filename.unique()
    #print(od_files)
    #od_dfs = {}
    #samples = {}
    #for od_file in od_files:
    #    od_dfs[od_file] = pd.read_csv(od_file)
        #samples[od_file] = od_dfs[od_file]['Sample_ID'].unique()
        
    dfs = join_meta_and_data(meta, od_files)
    
    return dfs

def join_meta_and_data(metadata_df, od_files):
    """
    Join based upon the "Sample_ID" in od_dfs and lab sample id in metadata_df
    """
    od_dfs = {}
    for od_df_file in od_files:
        od_df = pd.read_csv(od_df_file)
        od_df['Sample_ID'] = od_df['Sample_ID'].apply(lambda x: 'sample.transcriptic.' + x)
        od_df = od_df.merge(metadata_df, left_on='Sample_ID', right_on='sample_id', how='left') 
        od_dfs[od_df_file] = od_df

    return od_dfs

def get_all_sample_corrected_od_dataframe(od_files):
    df = pd.DataFrame()
    for out_file in od_files:
        if not os.path.isfile(out_file):
            continue
    
        plate_df = pd.read_csv(out_file, index_col=0)
        df = df.append(plate_df)
    df = df.loc[(df['strain'].notna())]
    return df

def merge_pre_post_od_dfs(first, second):
    """
    First has pre-dilution od, and second has final od
    """
    #print(first.columns)
    #print(second.columns)
    drop_cols = ['Sample', 'od', 'experiment_id', 'filename', 'lab', 'media',  'output','strain_circuit', 'strain_input_state', 'temperature']
    rename_map = {}
    for c in first.columns:
        if c not in ['experiment_id', 'filename', 'lab', 'media', 'od', 'output', 'replicate', 'strain', 'strain_circuit', 'strain_input_state', 'temperature']:
            rename_map[c] = 'pre_' + c
    first = first.drop(columns=drop_cols).rename(index=str, columns=rename_map)
    
    rename_map = {}
    for c in first.columns:
        if c not in ['experiment_id', 'filename', 'lab', 'media', 'od', 'output', 'replicate',  'strain', 'strain_circuit', 'strain_input_state', 'temperature']:
            rename_map[c] = 'post_' + c
    second = second.rename(index=str, columns=rename_map)
    
    return second.merge(first, on=['replicate', 'strain'], how='inner')

def make_etl_od_datafiles(data, out_dir):
    """
    Create one datafile per container that has corrected OD values.  Join
    datafiles for related containers into one for an experiment.
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for ex_id, od_files in data.items():
        for od_file in od_files:
            container = od_file.split('/')[-1].split('_')[0]
            out_file = os.path.join(out_dir, ex_id + '_' + str(container) + '.csv')
            if os.path.isfile(out_file):
                continue

            try:
                dfs = get_meta_and_etl_od_data(ex_id, [od_file])
                for k, v in dfs.items():
                    v.to_csv(out_file)
            except Exception as e:
                print("Could not process " + str(ex_id) + " because: " + str(e) )
                
    for ex_id, od_files in data.items():
        od_dfs = []
        out_file = os.path.join(out_dir, ex_id + '.csv')
        if os.path.isfile(out_file):
            continue
        if len(od_files) < 2:
            continue


    #    print(ex_id)
    #    print(od_files)
    #    assert(len(od_files) == 2)
        for od_file in od_files:
            container = od_file.split('/')[-1].split('_')[0]
            out_file = os.path.join(out_dir, ex_id + '_' + str(container) + '.csv')
            df = pd.read_csv(out_file, index_col=0)
            od_dfs.append(df)
        if len(od_dfs[0]) < len(od_dfs[1]):
            first = od_dfs[0]
            second = od_dfs[1]
        else:
            first = od_dfs[1]
            second = od_dfs[0]

        df = merge_pre_post_od_dfs(first, second)

        df.to_csv(out_file)


############################################################################
## Raw data functions available from pipeline
############################################################################

def make_raw_od_datafiles(ex_ids, out_dir):
    """
    Create one datafile per experiment that has uncorrected OD values.  
    """
    for ex_id in ex_ids:
        out_file = os.path.join(out_dir, ex_id + '.csv')
        if os.path.isfile(out_file):
            continue

        try:
            df = pipeline_od.get_experiment_od_data(ex_id)
            df.to_csv(out_file)
        except Exception as e:
            print("Could not process " + str(ex_id) + " because: " + str(e) )




def get_od_training_df(df):
    """
    Convert df into a dataframe suitable for regression.
    """
    df = df.rename(index=str, columns={'od_2': "output_od", 'od' : 'input_od'})
    #df = df[['od', 'media', 'temperature', 'strain', 'class_label']]
    df = df[['input_od', 'strain', 'output_od']]
    df = df.dropna(subset=['strain'])
    return df[['strain', 'output_od']], df['input_od']

def get_all_sample_od_dataframe(od_files):
    df = pd.DataFrame()
    for out_file in od_files:
        if not os.path.isfile(out_file):
            continue
    
        plate_df = pd.read_csv(out_file)
        df = df.append(plate_df)
    df = df.loc[(df['od_2'].notna()) & (df['strain'].notna())]
    return df

def get_experiment_od_and_metadata(metadata_df, od_dfs):
    #Strip leading sample id if applicable
    sample_ids = metadata_df.sample_id.unique()
    if len(sample_ids) > 0 and 'https' in sample_ids[0]:
        metadata_df['od_sample_id'] = metadata_df['sample_id'].apply(lambda x: ".".join(x.split('.')[2:]))
    elif len(sample_ids) > 0 and 'sample.transcriptic' in sample_ids[0]: 
        metadata_df['od_sample_id'] = metadata_df['sample_id'].apply(lambda x: x.split('.')[2])        
    else:
        metadata_df['od_sample_id'] = metadata_df['sample_id']

    #get collection id from od_dfs keys
    def get_collection_id_from_filename(filename):
        #print(filename)
        if 'od.csv' in filename:
            ## Assume filename has collection id in path
            collection_id = filename.split('/')[8]
            return collection_id
        else:
            ## Assume filename is of form od1.csv
            collection_id = filename.split('/')[-1].split('.')[0][2:]
            return collection_id
    
    collection_ids = {}
    for k,v in od_dfs.items():
        #print(k)
        collection_ids[get_collection_id_from_filename(k)] = k
    
    for k,v in collection_ids.items():
        #rename od_dfs columns with collection id index
        try:
            od_df = od_dfs[v].drop(columns=['container_id', 'aliquot_id'])
        except Exception as e:
            od_df = od_dfs[v]
        
        od_df.columns = [x + "_" + str(k) for x in od_df.columns]
        #print(metadata_df.shape)
        #print(od_df.columns)
        #print(metadata_df.columns)
        #print(k)
        
        metadata_df = metadata_df.merge(od_df, left_on='od_sample_id', right_on='sample'+ "_" + str(k), how='left') 
        metadata_df = metadata_df.drop(columns=["well_" + str(k), "sample_" + str(k)])
        #print(metadata_df.shape)
    metadata_df = metadata_df.drop(columns=['od_sample_id'])
    return metadata_df

def match_pre_post_od(df):
    """
    The dataframe has different sample ids for pre and post dilution samples.  The post
    samples are derived from pre samples, and this function adds the pre od for each sample 
    as a column for the corresponding post samples.
    """
    return df

def get_calibration_df(calibration_file):
    return pd.read_csv(calibration_file)






def get_strain_growth_plot(experiment_id):
    samples = pipeline.get_experiment_samples(experiment_id,file_type='CSV')
    calibration_samples = [x for x in samples if 'calibration' in x['filename']]
    ## Drop calibration_samples for now
    samples = [x for x in samples if x not in calibration_samples]
    
    #print(calibration_samples)
    #print("Got " + str(len(samples)) + " samples for " + experiment_id)
    
    
    
    meta = pipeline.get_metadata_dataframe(samples)
    od_files = meta.filename.unique()
    #print(od_files)
    od_dfs = {}
    for od_file in od_files:
        od_dfs[od_file] = pd.read_csv(od_file)
    df = get_experiment_od_and_metadata(meta, od_dfs)
    df = match_pre_post_od(df)
    return df

