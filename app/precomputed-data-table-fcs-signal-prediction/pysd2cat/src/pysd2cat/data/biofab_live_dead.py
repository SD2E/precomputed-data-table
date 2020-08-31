import pandas as pd
import os
import sys
import glob
import json
from os.path import expanduser
import matplotlib.pyplot as plt                   # For graphics
import numpy as np
from agavepy.agave import Agave, AgaveError
from agavepy.files.download import files_download
from pysd2cat.data import pipeline
from pysd2cat.analysis.Names import Names
from pysd2cat.data import tx_fcs
import json
import logging
l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


def make_experiment_metadata_dataframe(experiment_id):
    df = pd.DataFrame()
    samples = pipeline.get_experiment_samples(experiment_id,file_type='FCS')
    for sample in samples:
        df = df.append(sample_to_row(sample), ignore_index=True)
    return df

def flatten_sample_contents(feature, value):
    kv_pairs=[]
    for content in value:
        if "name" in content:
            if 'label' in content['name']:
                if content['name']['label'] == 'YPAD':
                    kv_pairs.append(('media', content['name']['label']))
                elif content['name']['label'] == 'Ethanol':
                    kv_pairs.append(('kill_method', 'Ethanol'))
                    if 'volume' in content:
                        if 'unit' in content['volume'] and 'value' in content['volume']:
                            kv_pairs.append(('kill_volume', content['volume']['value']))
                            kv_pairs.append(('kill_volume_unit', content['volume']['unit']))
                        else:
                            raise "Malformed Volume of sample contents: " + str(content['volume'])
                elif content['name']['label'] == 'SYTOX Red ':
                    #print(content['name'])
                    kv_pairs.append(('stain', 'SYTOX Red Stain'))
                    if 'volume' in content:
                        if 'unit' in content['volume'] and 'value' in content['volume']:
                            kv_pairs.append(('stain_volume', content['volume']['value']))
                            kv_pairs.append(('stain_volume_unit', content['volume']['unit']))
                        else:
                            raise "Malformed Volume of sample contents: " + str(content['volume'])
                    
    return kv_pairs

def flatten_temperature(feature, value):
    return [(feature, value['value'])]

def flatten_feature(feature, value):
    if feature == 'sample_contents':
        return flatten_sample_contents(feature, value)
    elif feature == 'temperature':
        return flatten_temperature(feature, value)
    else:
        raise Exception("Cannot flatten feature: " + feature)

def sample_to_row(sample):
    
    features = [k for k,v in sample.items() if type(v) is not dict and type(v) is not list]
    row = {}
    for feature in features:
        row[feature] = sample[feature]
        
    #Handle nested features
    for feature in [x for x in sample.keys() if x  not in features]:
        try:
            kv_pairs = flatten_feature(feature, sample[feature])
            for k, v in kv_pairs:
                row[k] = v
        except Exception as e:
            pass
    return row

def fetch_data(meta_df, data_dir, overwrite=False):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    ag = Agave.restore()

    for i, row in meta_df.iterrows():
        src = row['agave_system'] + row['agave_path']
        dest = data_dir + row['agave_path']

        if overwrite or not os.path.exists(dest):
            result_dir = "/".join((data_dir + row['agave_path']).split('/')[0:-1])
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            #print(src)
            print(dest)
            
            files_download(ag.api_server, ag.token.token_info['access_token'], src, dest)
    
def write_meta_and_data_dataframe(experiment, overwrite=False):
    data_dir = os.path.join('data/biofab', experiment)
    all_data_file = os.path.join(data_dir, 'data.csv')

    if overwrite or not os.path.exists(all_data_file):
        meta_df = make_experiment_metadata_dataframe(experiment)
        fetch_data(meta_df, data_dir)
        meta_df[Names.FILENAME] = meta_df.apply(lambda x:  data_dir + "/" + x['agave_path'], axis=1)
        all_data_df = pipeline.get_data_and_metadata_df(meta_df, '.', fraction=None, max_records=None)
        all_data_df.to_csv(all_data_file)

def get_experiment_data_df(experiment):
    data_dir = os.path.join('data/biofab', experiment)
    all_data_file = os.path.join(data_dir, 'data.csv')
    experiment_df = pd.read_csv(all_data_file, index_col=0)
    return experiment_df

def get_flow_plate_properties(run_id, transcriptic_email, transcriptic_token, work_dir, aliquot_map_technique, download_data=False, container="Flow (Sytox) Plate 1"):
    print("Getting data for container: " + str(container))
    fcs_path = os.path.join(work_dir, container.replace(" ", "_"))
    if not os.path.exists(fcs_path):
        os.makedirs(fcs_path)
        download_data = True
    container_name = container.replace('(', '\(').replace(')', '\)')
    file_info = tx_fcs.create_fcs_manifest_and_get_files(run_id, 
                                                         transcriptic_email, 
                                                         transcriptic_token,
                                                         fcs_path=fcs_path, 
                                                         download_zip=download_data,
                                                         data_set_name=container_name,
                                                         container_name=container_name
                                                         )
    file_info['container'] = container

    if aliquot_map_technique == "titration1":
        #Flow plate didn't have the properties, so need to get them from another plate
        inoc_plate_info = tx_fcs.get_plate_well_properties(tx_fcs.get_tx_run(run_id), 
                                                       container_name="Inoculation Plate")

        for well_id, info in file_info['aliquots'].items():
            info['properties'] = inoc_plate_info[well_id.upper()]
            
    return file_info

def make_row(well, info, aliquot_map_technique=None):
    row = { 
        "well" : well,
#        "filename" : info['file'],
#        "checksum" : info['checksum']
        }
    if 'file' in info:
        row['filename'] = info['file']
    if 'checksum' in info:
        row['checksum'] = info['checksum']
    # Properties don't include sytox, so add manually
    def well_stain(well):
        if 'a' in well or 'b' in well or 'c' in well or 'd' in well:
            return "SYTOX Red Stain"
        else:
            return None
        
    row['stain'] = well_stain(well) 
    
    def expand_options(v):
        #print(v)
        kv_pairs = {}
        for option, value in v.items():
            if option == 'Reagents':
                #print(option)
                for reagent, properties in value.items():
                    if reagent == 'Ethanol':
                        kv_pairs['kill'] = reagent
                        kv_pairs['kill_volume'] = properties['final_concentration']['qty']
        return kv_pairs
    
    if aliquot_map_technique == 'fifteen_zero':
        row['kill'] = 'Ethanol'
        killed = '9' not in well
        row['kill_volume'] = 300.0 if killed else 0.0 
    elif aliquot_map_technique == "titration_1":
        if 'properties' in info:
            for k, v in info['properties'].items():
                if k == 'Options':
                    row.update(expand_options(ast.literal_eval(v)))
                else:
                    row[k] = str(v)
    elif '10to80' in aliquot_map_technique:
        row['kill'] = 'Ethanol'
        if '5' in well:
            row['kill_volume'] = 140.0
        elif '6' in well:
            row['kill_volume'] = 210.0
        elif '7' in well:
            row['kill_volume'] = 280.0
        elif '8' in well:
            row['kill_volume'] = 1120.0
        else:
            row['kill_volume'] = 0.0


        

                
                
                
    return row

def make_meta_dataframe(run_id, plate_properties, aliquot_map_technique=None):
    data_df = pd.DataFrame()
    for well, info in plate_properties['aliquots'].items():
        row = make_row(well, info, aliquot_map_technique=aliquot_map_technique)
        data_df = data_df.append(row, ignore_index=True)
        
    data_df['experiment_id'] = run_id
    return data_df
        
def get_container_work_dir(out_dir, container):
    return os.path.join(out_dir, container.replace(" ", "_"))

def get_container_data(out_dir, container):
    path = os.path.join(get_container_work_dir(out_dir, container), 'data.csv')
    df = pd.read_csv(path)
    return df

def make_container_dataframes(run_id, plate_properties_list, out_dir, aliquot_map_technique=None, overwrite=False):
    for plate_properties in plate_properties_list:
        df_path = get_container_work_dir(out_dir, plate_properties['container'])
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        df_file = os.path.join(df_path, 'data.csv')
        if overwrite or not os.path.exists(df_file):
            meta_df = make_meta_dataframe(run_id, plate_properties, aliquot_map_technique=aliquot_map_technique)
            df = pipeline.get_data_and_metadata_df(meta_df, '.', fraction=None, max_records=30000)
            # drop bead controls in column 12
            df = df.loc[~df['well'].str.contains('12')]
            df.to_csv(df_file)
            
def extract_time_point(x):
    return x.Name.split(' ')[3].split('_')[0]
def extract_time(x):
    return x.Name.split(' ')[3].split('_')[1]
def assign_container(prefix, x, containers):
    name = prefix + ' ' + x.time_point
    #print(containers)
    return next(c for c in containers.Containers if name in c.attributes['label'])

def get_flow_dataframe(run_id):
    """
    Get a dataframe that has the time and timepoints of each dataset
    """
    run_obj = tx_fcs.get_tx_run(run_id)
    d = run_obj.data

    try:
        containers = run_obj.containers
    except Exception as e:
        containers = run_obj.containers

    flow_data = d.loc[d.Operation == 'flow_analyze']

    if len(flow_data) > 0:
        flow_data.loc[:,'time_point'] = flow_data.apply(extract_time_point, axis=1)
        flow_data.loc[:,'time'] = flow_data.apply(extract_time, axis=1)
        flow_data.loc[:, 'container'] = flow_data.apply(lambda x: assign_container('Flow (Sytox) Plate', x, containers), axis=1)
        flow_data.loc[:, 'container_name'] = flow_data.apply(lambda x: x['container'].attributes['label'], axis=1)
        flow_data['time_point'] = flow_data['time_point'].astype('int64')
        flow_data['time'] = flow_data['time'].astype('float')

    return flow_data

def write_flow_data_and_metadata(run_id, tx_config, aliquot_map_technique, work_dir='.', overwrite=False):
    """
    Writes dataframe for each plate that has all the data and metadata.

    Returns:
         flow_data (a dataframe with dataset info about containers and timepoints)
         plate_properties (a list of dicts describing each plate and wells)
    """
    flow_data_path = os.path.join(work_dir,'flow_data.csv')
    plate_properties_path = os.path.join(work_dir,'plate_properties.json')

    #If you have already fetched data from TX and stored it, just read it in locally
    if os.path.exists(flow_data_path) & os.path.exists(plate_properties_path):
        flow_data = pd.read_csv(flow_data_path)
        with open(plate_properties_path,'r') as f:
            plate_properties = json.load(f)
    #Otherwise fetch from TX
    else:
        transcriptic_email = tx_config['email']
        transcriptic_token = tx_config['token']


        flow_data = get_flow_dataframe(run_id)
        plate_properties = [ get_flow_plate_properties(run_id,
                                                           transcriptic_email,
                                                           transcriptic_token,
                                                           work_dir,
                                                           aliquot_map_technique,
                                                           download_data=overwrite,
                                                           container=flow_data.container[i].attributes['label']) \
                                 for i in flow_data.index]
        make_container_dataframes(run_id, plate_properties, work_dir, aliquot_map_technique=aliquot_map_technique, overwrite=overwrite)

        flow_data.to_csv(flow_data_path)
        with open(plate_properties_path,'w') as f:
            json.dump(plate_properties,f)


    return flow_data, plate_properties

