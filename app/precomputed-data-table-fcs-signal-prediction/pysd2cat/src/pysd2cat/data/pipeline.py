import pymongo
import json
import pandas as pd
import os
import FlowCytometryTools as FCT
from pysd2cat.analysis.Names import Names
import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


dbURI = 'mongodb://readonly:WNCPXXd8ccpjs73zxyBV@catalog.sd2e.org:27020/admin?readPreference=primary'
client = pymongo.MongoClient(dbURI)
db = client.catalog_staging
science_table=db.science_table
jobs_table=db.jobs

###############################################
# BioFab Live Dead Related Data Gathering     #
###############################################



###############################################
# Helpers for building a live/dead classifier #
###############################################


def get_dataframe_for_live_dead_classifier(data_dir,control_type=[Names.WT_DEAD_CONTROL, Names.WT_LIVE_CONTROL],fraction=None, max_records=None):
    """
    Get pooled FCS data for every live and dead control. 
    """

    meta_df = get_metadata_dataframe(get_live_dead_controls(control_type))
    print("meta_df")
    print(meta_df.columns.tolist())
    print(meta_df.head(5))
    #meta_df.to_csv("metadata_before_masking.csv")

    ##Drop columns that we don't need
    da = meta_df[[Names.STRAIN, Names.FILENAME]].copy()
    da[Names.STRAIN] = da[Names.STRAIN].mask(da[Names.STRAIN] == Names.WT_DEAD_CONTROL,  0)
    da[Names.STRAIN] = da[Names.STRAIN].mask(da[Names.STRAIN] == Names.WT_LIVE_CONTROL,  1)
    da = da.rename(index=str, columns={Names.STRAIN: "class_label"})
    print("da before get_data_and_metadata_df")
    print(da.columns.tolist())
    print(da.head(5))
    da = get_data_and_metadata_df(da, data_dir,fraction,max_records)
    print("da after get_data_and_metadata_df")
    print(da.columns.tolist())
    print(da.head(5))
    da = da.drop(columns=[Names.FILENAME, 'Time'])
    return da

def get_calibration_samples(experiment_id, calibration_type=[Names.LUDOX]):
    query={}
    query[Names.CHALLENGE_PROBLEM] = Names.YEAST_STATES
    query[Names.FILE_TYPE] = Names.CSV
    query[Names.STANDARD_TYPE] = {"$in": calibration_type}
    #print("Query:")
    #print(query)
    results = []
    #print("Printing results of query...")
    for match in science_table.find(query):
        #print(match)
        match.pop('_id')
        results.append(match)
    #print("Printing length of results...")
    #print(len(results))
    return results


    samples = pipeline.get_experiment_samples(experiment_id,file_type='CSV')
    calibration_samples = [x for x in samples if 'calibration' in x['filename']]
    return calibration_samples



def get_live_dead_controls(control_type=[Names.WT_DEAD_CONTROL, Names.WT_LIVE_CONTROL]):
    """
    Get metadata for every live and dead control sample across
    all experiments.
    """
    query={}
    query[Names.CHALLENGE_PROBLEM] = Names.YEAST_STATES
    query[Names.FILE_TYPE] = Names.FCS
    query[Names.STRAIN] = {"$in": control_type}
    #print("Query:")
    #print(query)
    results = []
    print("Printing results of query...")
    for match in science_table.find(query):
        #print(match)
        match.pop('_id')
        results.append(match)
    print("Printing length of results...")
    print(len(results))
    return results

def get_experiment_jobs(experiment_id, gating = 'auto', status='FINISHED'):
    """
    Get job info related to experiment
    """
    query={}
    query['data.sample_id']=experiment_id
    query['data.message.gating'] = gating
    query['status']={"$in" : ["FINISHED", "VALIDATED"]}
    matches=list(jobs_table.find(query))
    return matches

def get_experiment_samples(experiment_id, file_type='FCS'):
    """
    Get metadata for every live and dead control sample across
    all experiments.
    """
    query={}
    query['experiment_id'] = experiment_id
    query['file_type'] = file_type

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    return results


strain_inputs = {
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7376/1' : {'gate' : 'AND', 'input' : '00','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7375/1' : {'gate' : 'AND', 'input' : '01','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7373/1' : {'gate' : 'AND', 'input' : '10','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7374/1' : {'gate' : 'AND', 'input' : '11','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8544/1' : {'gate' : 'NAND', 'input' : '00','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8545/1' : {'gate' : 'NAND', 'input' : '01','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8543/1' : {'gate' : 'NAND', 'input' : '10','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8542/1' : {'gate' : 'NAND', 'input' : '11','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6390/1' : {'gate' : 'NOR', 'input' : '00','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6388/1' : {'gate' : 'NOR', 'input' : '01','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6389/1' : {'gate' : 'NOR', 'input' : '10','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6391/1' : {'gate' : 'NOR', 'input' : '11','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8225/1' : {'gate' : 'OR', 'input' : '00','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_5993/1' : {'gate' : 'OR', 'input' : '01','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_5783/1' : {'gate' : 'OR', 'input' : '10','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_5992/1' : {'gate' : 'OR', 'input' : '11','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7300/1' : {'gate' : 'XNOR', 'input' : '00','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8231/1' : {'gate' : 'XNOR', 'input' : '01','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7377/1' : {'gate' : 'XNOR', 'input' : '10','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7299/1' : {'gate' : 'XNOR', 'input' : '11','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16970/1' : {'gate' : 'XOR', 'input' : '00','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16969/1' : {'gate' : 'XOR', 'input' : '01','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16968/1' : {'gate' : 'XOR', 'input' : '10','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16967/1' : {'gate' : 'XOR', 'input' : '11','output' : '0'},
    
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7376/1' : {'gate' : 'AND', 'input' : '00','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7375/1' : {'gate' : 'AND', 'input' : '01','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7373/1' : {'gate' : 'AND', 'input' : '10','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7374/1' : {'gate' : 'AND', 'input' : '11','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8544/1' : {'gate' : 'NAND', 'input' : '00','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8545/1' : {'gate' : 'NAND', 'input' : '01','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8543/1' : {'gate' : 'NAND', 'input' : '10','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8542/1' : {'gate' : 'NAND', 'input' : '11','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6390/1' : {'gate' : 'NOR', 'input' : '00','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6388/1' : {'gate' : 'NOR', 'input' : '01','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6389/1' : {'gate' : 'NOR', 'input' : '10','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_6391/1' : {'gate' : 'NOR', 'input' : '11','output' : '0'},
            'UWBF_OR_00' : {'gate' : 'OR', 'input' : '00','output' : '0'},
            'UWBF_OR_01' : {'gate' : 'OR', 'input' : '01','output' : '1'},
            'UWBF_OR_10' : {'gate' : 'OR', 'input' : '10','output' : '1'},
            'UWBF_OR_11' : {'gate' : 'OR', 'input' : '11','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7300/1' : {'gate' : 'XNOR', 'input' : '00','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_8231/1' : {'gate' : 'XNOR', 'input' : '01','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7377/1' : {'gate' : 'XNOR', 'input' : '10','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_7299/1' : {'gate' : 'XNOR', 'input' : '11','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16970/1' : {'gate' : 'XOR', 'input' : '00','output' : '0'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16969/1' : {'gate' : 'XOR', 'input' : '01','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16968/1' : {'gate' : 'XOR', 'input' : '10','output' : '1'},
            'https://hub.sd2e.org/user/sd2e/design/UWBF_16967/1' : {'gate' : 'XOR', 'input' : '11','output' : '0'}
        }
    
media_map = {
    'https://hub.sd2e.org/user/sd2e/design/culture_media_4/1' : 'SC Media',
    'https://hub.sd2e.org/user/sd2e/design/culture_media_5/1' : 'YPAD',
    'https://hub.sd2e.org/user/sd2e/design/culture_media_3/1' : 'SC High Osm',
    'https://hub.sd2e.org/user/sd2e/design/culture_media_2/1' : 'SC Slow'
}

def handle_missing_data(result, key):
    if Names.STRAIN in result: 
        strain = str(result[Names.STRAIN])
    else:
        strain = None
      

        
    if key in Names.STRAIN_INPUT_STATE and strain is not None and strain in strain_inputs:
        return strain_inputs[strain]['input']
    if key in Names.INPUT and strain is not None and strain in strain_inputs:
        return strain_inputs[strain]['input']
    elif key in Names.STRAIN_CIRCUIT and strain is not None  and strain in strain_inputs:
        return strain_inputs[strain]['gate']
    elif key in Names.GATE and strain is not None  and strain in strain_inputs:
        return strain_inputs[strain]['gate']
    elif key in Names.OUTPUT and strain is not None  and strain in strain_inputs:
        return strain_inputs[strain]['output']
    elif key in 'media':
        return media_map[result['media']]
    else:
        return None



def get_metadata_dataframe(results):
    """
    Convert science table results into metadata dataframe.
    """
    runtime = detect_runtime()
    print("runtime: {}".format(runtime))    
    meta_df = pd.DataFrame()
    for result in results:
               
        result_df = {}
        keys_to_set = [Names.STRAIN, Names.FILENAME, Names.LAB, Names.SAMPLE_ID,
                       Names.STRAIN_CIRCUIT, Names.STRAIN_INPUT_STATE,
                       Names.EXPERIMENT_ID, Names.REPLICATE, Names.OUTPUT
                      ]
        for k in keys_to_set:
            if k in result:
                result_df[k] = result[k]
            else:
                result_df[k] = handle_missing_data(result, k)
        if 'jupyter' in runtime:
            result_df['filename'] = result['jupyter_path']
        else:
            result_df['filename'] = result['hpc_path']

        if not os.path.exists(result_df[Names.FILENAME]):
            # Fix error where wrong path exists with `uploads`
            if 'uploads' in result_df[Names.FILENAME]:
                result_df[Names.FILENAME] = result_df[Names.FILENAME].replace('uploads/', '')               
            else:
                continue

        ## Other values (not at top level)
        if Names.INOCULATION_DENSITY in result:
            result_df['od'] = result[Names.INOCULATION_DENSITY]['value']
        else:
            result_df['od'] = None
        if 'sample_contents' in result:
            result_df['media'] = result['sample_contents'][0]['name']['label']
            if 'https://hub.sd2e.org/user/sd2e/design' in result_df['media']:
                result_df['media'] = handle_missing_data(result_df, 'media')
        else:
            result_df['media'] = None
            
        if 'temperature' in result:
            result_df['temperature'] = result['temperature']['value']
        else:
            result_df['temperature'] = None
            
        if Names.STRAIN_CIRCUIT in result_df and Names.STRAIN_INPUT_STATE in result_df:
            result_df[Names.OUTPUT] = gate_output(result_df[Names.STRAIN_CIRCUIT], result_df[Names.STRAIN_INPUT_STATE])
        else:
            result_df[Names.OUTPUT] = None

        #result_df['fcs_files'] = ['agave://' + result['agave_system'] + result['agave_path']]

#        try:
#            result_ex_id = result['sample_id'].split("/")[0].split(".")[-1]
#        except Exception as e:
#            result_ex_id = None
#        result_df['ex_id'] = result_ex_id


        meta_df = meta_df.append(result_df, ignore_index=True)
    #pd.set_option('display.max_colwidth', -1)
    #print("Printing metadata df")
    #print(meta_df)
    return meta_df

def detect_runtime():
    if 'REACTORS_VERSION' in os.environ:
        return 'abaco'
    elif 'JUPYTERHUB_USER' in os.environ:
        return 'jupyter'
    elif 'TACC_DOMAIN' in os.environ:
        return 'hpc'
    else:
#        raise Exception('Not a known runtime')
        print("os.environ: {}".format(os.environ))
        return 'cli'

def get_flow_dataframe(data_dir, filename, logger=l):
    try:
        df = FCT.FCMeasurement(ID=filename, datafile=os.path.join(data_dir, filename)).read_data()
    except Exception as e:
        logger.warn("Problem with FCS file: " + str(filename))
        
    return df


def get_data_and_metadata_df(metadata_df, data_dir, fraction=None, max_records=None, logger=l):
    """
    Join each FCS datatable with its metadata.  Costly!
    """
    #dataset_local_df=pd.DataFrame()
    all_data_df = pd.DataFrame()
    for i, record in metadata_df.iterrows():
        ## Substitute local file for SD2 URI to agave file 
        #record['fcs_files'] = local_datafile(record['fcs_files'][0], data_dir)
        #dataset_local_df = dataset_local_df.append(record)
        logger.debug("record[{}]: {}".format(Names.FILENAME, record[Names.FILENAME]))
        if not os.path.exists(record[Names.FILENAME]):
            # Fix error where wrong path exists with `uploads`
            if 'uploads' in record[Names.FILENAME]:
                record[Names.FILENAME] = record[Names.FILENAME].replace('uploads/', '')    
                if not os.path.exists(record[Names.FILENAME]):
                    continue
            else:
                continue
    
        ## Create a data frame out of FCS file
        data_df = get_flow_dataframe(data_dir,record[Names.FILENAME], logger=logger)

        if max_records is not None:
            data_df = data_df[0:min(len(data_df), max_records)]
        elif fraction is not None:
            logger.debug("In get_data_metadata_df ELIF condition fraction is: ", fraction)
            data_df = data_df.sample(frac=fraction, replace=True)
            #data_df = data_df.replace([np.inf, -np.inf], np.nan)
        #data_df = data_df[~data_df.isin(['NaN', 'NaT']).any(axis=1)]

        data_df[Names.FILENAME] = record[Names.FILENAME]
        all_data_df = all_data_df.append(data_df)

    ## Join data and metadata
    final_df = metadata_df.merge(all_data_df, left_on='filename', right_on='filename', how='inner')
    return final_df

def sanitize(my_string):
    return my_string.replace('-', '_').replace(' ', '_')

def get_xplan_data_and_metadata_df(metadata_df, data_dir, fraction=None, max_records=None, strain_col='strain', logger=l):
    """
    Rename columns from data and metadata to match xplan columns
    """
    logger.info("Getting xplan dataframe for experiment...")
    df = get_data_and_metadata_df(metadata_df, data_dir, fraction=fraction, max_records=max_records, logger=logger)
    logger.info("Renaming columns ...")
    rename_map = {
#        "experiment_id" : "plan",
#        "sample_id" : "id",
        "strain_input_state" : "input_state",
        "strain_circuit" : "gate",
#        "strain_sbh_uri" : "strain",
        "strain" : strain_col
#        "temperature" : 'inc_temp'
    }
    final_rename_map = {}
    for col in df.columns:
        if col not in rename_map:
            final_rename_map[col] = sanitize(col)
    
    for col in rename_map:
        if col in df.columns:
            final_rename_map[col] = rename_map[col]
    logger.info("renaming columns as: " + str(final_rename_map))
    logger.info("columns: " + str(df.columns))
    try:
        df = df.rename(columns=final_rename_map)
    except Exception as e:
        logger.info(e)
    logger.info("Renamed columns ...")
    return df
    
def get_mefl_data_and_metadata_df(metadata_df, fraction=None, max_records=None):
    """
    Join each FCS datatable with its metadata.  Costly!
    """
    
    ex_id = metadata_df[Names.EXPERIMENT_ID].unique()[0]
    results = get_experiment_jobs(ex_id)
    
    if len(results) == 0:
        raise Exception("No MEFL results for: " + ex_id)
    
    output = [ x['UPDATE']['data']['outputs']['bayesdb_data'] for x in results[0]['history'] if 'UPDATE' in x and 'data' in x['UPDATE'] and 'outputs' in x['UPDATE']['data'] and 'bayesdb_data' in  x['UPDATE']['data']['outputs']][0]
    print(output)
    runtime = detect_runtime()
    if runtime is 'jupyter':
        myfile = os.path.join('/home/jupyter/sd2e-community', output.split('data-sd2e-community')[1][1:])
    else:
        myfile = os.path.join('/work/projects/SD2E-Community/prod/data', output.split('data-sd2e-community')[1][1:])
    df = pd.read_csv(myfile, memory_map=True)   
    df=df.drop(columns=['strain', 'replicate'])
    ## Join data and metadata
    final_df = metadata_df.merge(df, left_on=Names.SAMPLE_ID, right_on=Names.SAMPLE_ID, how='outer')    
    return final_df
    
def get_xplan_mefl_data_and_metadata_df(metadata_df, fraction=None, max_records=None):
    """
    Rename columns from data and metadata to match xplan columns
    """
    df = get_mefl_data_and_metadata_df(metadata_df, fraction=fraction, max_records=max_records)
    rename_map = {
        "experiment_id" : "plan",
        "sample_id" : "id",
        "strain_input_state" : "input",
        "strain_circuit" : "gate",
        "strain_sbh_uri" : "strain",
        "strain" : "strain_name",
        "temperature" : 'inc_temp'
    }
    #for col in df.columns:
    #    if col not in rename_map:
    #        rename_map[col] = sanitize(col)
    #print("renaming columns as: " + str(rename_map))
    df = df.rename(index=str, columns=rename_map)
    return df    
    

def get_mefl_histograms_and_metadata_df(metadata_df, results):
    """
    Join each FCS datatable with its metadata.  Costly!
    """
    
    
    dfs = []
    
    for result in results:
        #print("Processing result: " + str(result))
        completion_tag = 'UPDATE'
        output = [ x[completion_tag]['data']['outputs']['output'] for x in result['history'] if completion_tag in x and 'data' in x[completion_tag] and 'outputs' in x[completion_tag]['data'] and 'output' in  x[completion_tag]['data']['outputs']]
        #print(output)
        if len(output) == 0:
            continue
            
        output=output[0]    
        runtime = detect_runtime()
        if runtime is 'jupyter':
            myfile = os.path.join('/home/jupyter/sd2e-community', output.split('data-sd2e-community')[1][1:])
        else:
            myfile = os.path.join('/work/projects/SD2E-Community/prod/data', output.split('data-sd2e-community')[1][1:])
        df = pd.read_csv(myfile)   

        df=df.drop(columns=['strain', 'replicate'])
        #for col in df.columns[8:]:
        #    df[col] = df[col].astype(float)
        #print(df.head())
        #print(metadata_df.head())
        ## Join data and metadata
        final_df = metadata_df.merge(df, left_on=Names.SAMPLE_ID, right_on=Names.SAMPLE_ID, how='outer')    
        final_df['session'] = result['session']
        dfs.append(final_df)
    return dfs

def get_xplan_mefl_histograms_and_metadata_df(metadata_df, results):
    """
    Rename columns from data and metadata to match xplan columns
    """
    
    

    dfs = get_mefl_histograms_and_metadata_df(metadata_df, results)
    #print(dfs)
    rename_map = {
        "experiment_id" : "plan",
        "sample_id" : "id",
        "strain_input_state" : "input",
        "strain_circuit" : "gate",
        "strain_sbh_uri" : "strain",
        "strain" : "strain_name",
        "temperature" : 'inc_temp'
    }
    my_dfs = []
    for df in dfs:
        #print("Sanitizing result " + str(df))
        for col in df.columns:
            if col not in rename_map:
                rename_map[col] = sanitize(col)
        #print("renaming columns as: " + str(rename_map))
        df = df.rename(index=str, columns=rename_map)

        df['replicate'] = df['replicate'].fillna(0).astype(int)
        my_dfs.append(df)
    return my_dfs      
    
###############################################
# Helpers for getting sample data to classify #
###############################################

def gate_output(gate, inputs):
    if gate == Names.NOR:
        if inputs == Names.INPUT_00:
            return 1
        else:
            return 0
    elif gate == Names.AND:
        if inputs == Names.INPUT_11:
            return 1
        else:
            return 0
    elif gate == Names.NAND:
        if inputs == Names.INPUT_11:
            return 0
        else:
            return 1
    elif gate == Names.OR:
        if inputs == Names.INPUT_00:
            return 0
        else:
            return 1
    elif gate == Names.XOR:
        if inputs == Names.INPUT_00 or inputs == Names.INPUT_11:
            return 0
        else:
            return 1
    elif gate == Names.XNOR:
        if inputs == Names.INPUT_00 or inputs == Names.INPUT_11:
            return 1
        else:
            return 0

def get_strain_dataframe_for_classifier(circuit, input, od=0.0003, media='SC Media', experiment='', data_dir='', fraction=None):
    """
    """
    print("inside get_strain_dataframe_for_classifier")
    print("circuit: {} input: {} od: {} media: {} experiment: {}".format(circuit, input, od, media, experiment))
    results = get_strain(circuit, input, od=od, media=media, experiment=experiment)
    da = None
    if results:
        meta_df = get_metadata_dataframe(results)
        print(meta_df.columns)
        da = meta_df[[Names.FILENAME, Names.SAMPLE_ID, 'output']].copy()
        #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Dead-Control',  0)
        #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Live-Control',  1)
        da = get_data_and_metadata_df(da, data_dir, fraction=fraction)
        #da = da.drop(columns=['fcs_files', 'Time', 'RL1-W', 'BL1-W'])
        da = da.drop(columns=['filename', 'Time'])
        print("da")
        print(da.head(5))
    
    return da


def get_strain(strain_circuit, strain_input_state,od=0.0003, media='SC Media',experiment=''):
    query={}
    query[Names.CHALLENGE_PROBLEM] = Names.YEAST_STATES
    query[Names.FILE_TYPE] = Names.FCS
    query[Names.LAB] = Names.TRANSCRIPTIC
    
#    if strain_circuit == 'XOR' and strain_input_state == '00':
#        query['strain'] = '16970'
#    else:
    query[Names.STRAIN_CIRCUIT] = strain_circuit
    query[Names.STRAIN_INPUT_STATE] = strain_input_state
    query[Names.INOCULATION_DENSITY_VALUE] =  od
    query['sample_contents.0.name.label'] =  media

    #    query['experiment_id'] = experiment
    #query['filename'] = { "$regex" : ".*" + experiment +".*"}
    
    #query['strain'] = {"$in": ['WT-Dead-Control', 'WT-Live-Control']}

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    return results


def get_sample(sample_id):
    query={}
    query['sample_id'] = sample_id
    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    return results

def get_sample_time(sample):
    sample_id = sample['id']
    #print("Getting time for: " + sample_id)
    if 'transcriptic' in sample_id:
        sample_results = get_sample(sample_id)
    else:
        sample_results = []
        
    if len(sample_results) == 0:
        return None #"2019_04_01_12_00_00"
        #raise Exception("No sample results for: " + str(sample_id))
    elif len(sample_results) > 1:
        return sample_results[0]['created']
        #raise Exception("Multiple results for: " + str(sample_id))
    else:
        time = sample_results[0]['created']
        return time #time.strftime("%Y_%m_%d_%H_%M_%S")


def get_control(circuit, control, od=0.0003, media='SC Media'):
    query={}
    query['challenge_problem'] = 'YEAST_STATES'
    query['file_type'] = 'FCS'
    query['lab'] = 'Transcriptic'
    #query['strain_circuit'] = strain_circuit
    #query['strain_input_state'] = strain_input_state
    query['inoculation_density.value'] =  od
    query['sample_contents.0.name.label'] =  media
    
    query['strain'] = control

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    dumpfile = "strains.json"
    json.dump(results,open(dumpfile,'w'))
    return results

def get_experiment_ids(challenge_problem='YEAST_STATES', lab='Transcriptic'):
    query={}
    query['challenge_problem'] = challenge_problem
    query['lab'] = lab

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    
    experiment_ids = list(frozenset([result['experiment_id'] for result in results]))
    
    return experiment_ids

def get_experiment_strains(experiment_id):
    query={}
    query['experiment_id'] = experiment_id

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    
    strains = list(frozenset([result['strain'] for result in results if 'strain' in result]))
   
    return strains

def get_fcs_measurements(strain, experiment_id):
    query={}
    query['strain'] = strain
    query['experiment_id'] = experiment_id
    query['file_type'] = 'FCS'


    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    #print(results)
    files = list(frozenset([result['filename'] for result in results if 'filename' in result]))
   
    #result['agave_system'] + result['agave_path']
    return files





def get_control_dataframe_for_classifier(circuit, control, od=0.0003, media='SC Media', data_dir=''):
    meta_df = get_metadata_dataframe(get_control(circuit, control, od=od, media=media))
    da = meta_df[['fcs_files']].copy()
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Dead-Control',  0)
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Live-Control',  1)
    da = get_data_and_metadata_df(da, data_dir)
    #da = da.drop(columns=['fcs_files', 'Time', 'RL1-W', 'BL1-W'])
    da = da.drop(columns=['fcs_files', 'Time'])
    return da



## Convert agave path to local path
def local_datafile(datafile, data_dir):
    local_datafile = os.path.join(*(datafile.split('/')[3:]))   
    local_name = os.path.join(data_dir, local_datafile)
    print(local_name)
    return local_name


## Get one channel from FCS file
def get_measurements(fcs_file, channel, data_dir):
    #print("get_measurements: " + fcs_file + " " + channel)
    fct = FCT.FCMeasurement(ID=fcs_file, datafile=local_datafile(fcs_file, data_dir))
    return fct.data[[channel]]
    
