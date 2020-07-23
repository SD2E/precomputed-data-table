import json
import os
import pandas as pd

## Data Helper functions

def get_record(experiment):
    record = json.load(open(os.path.join(experiment, "record.json")))
    return record

def json_to_pd(json_data, channels=["FSC-A"]):    
    df = pd.DataFrame()
    for sample in json_data:
        sample_id = sample['sample_id']       
        sample_df = pd.DataFrame(data={ k:v for k, v in sample.items() if k != "sample_id"})        
        sample_df.loc[:,'sample_id'] = sample_id
        df = df.append(sample_df, ignore_index=True)
    return df
            
def get_record_file(record, file_type="fc_meta"):
    files = record['files']
    files_of_type = [ x for x in files if file_type in x['name']]
    if len(files_of_type) > 0:
        return files_of_type[0]
    else:
        return None

def get_meta(experiment, record):
    meta_file_name = get_record_file(record, file_type="fc_meta.csv")
    #print(meta_file_name)
    if meta_file_name:
        meta_df = pd.read_csv(os.path.join(experiment, meta_file_name['name']))
        return meta_df
    else:
        return None
    

def get_data(experiment, record):
    fc_raw_file = get_record_file(record, file_type="fc_raw_events")
    if fc_raw_file:
        fc_raw_data = json.load(open(os.path.join(experiment, fc_raw_file['name'])))
        return json_to_pd(fc_raw_data)
    else:
        return None

def get_data_and_metadata(experiment):
    record = get_record(experiment)
    data = get_data(experiment, record)
    meta = get_meta(experiment, record)
    if data is not None and meta is not None:
        df = meta.merge(data, on="sample_id", how="inner")
        return df
    else:
        return None
    
def get_data_converge_id(path):
    data_converge_id = path.split("/")[-1].split(".")[0]
    return data_converge_id


