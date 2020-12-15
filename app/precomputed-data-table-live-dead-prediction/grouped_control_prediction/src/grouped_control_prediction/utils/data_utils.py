import json
import os
import numpy as np
import pandas as pd


## Data Helper functions

def get_record(experiment):
    record = json.load(open(os.path.join(experiment, "record.json")))
    return record


def json_to_pd(json_data, channels=["FSC-A"]):
    df = pd.DataFrame()
    for sample in json_data:
        sample_id = sample['sample_id']
        sample_df = pd.DataFrame(data={k: v for k, v in sample.items() if k != "sample_id"})
        sample_df.loc[:, 'sample_id'] = sample_id
        df = df.append(sample_df, ignore_index=True)
    return df


def get_record_file(record, file_type="fc_meta"):
    files = record['files']
    files_of_type = [x for x in files if file_type in x['name']]
    if len(files_of_type) > 0:
        return files_of_type[0]
    else:
        return None


def get_meta(experiment, record):
    meta_file_name = get_record_file(record, file_type="fc_meta.csv")
    # print(meta_file_name)
    if meta_file_name:
        meta_df = pd.read_csv(os.path.join(experiment, meta_file_name['name']))
        return meta_df
    else:
        return None


def explode(df, lst_cols, fill_value=''):
    # taken from https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe
    # this function explodes a single row into multiple rows, by transforming one column value into multiple values,
    # each becoming a new row with the value of the other columns preserved.

    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols}) \
                   .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col: np.concatenate(df[col].values) for col in lst_cols}) \
                   .append(df.loc[lens == 0, idx_cols]).fillna(fill_value) \
                   .loc[:, df.columns]


def get_data_mem(experiment, record):
    fc_raw_file = get_record_file(record, file_type="fc_raw_events")
    if fc_raw_file:
        # get file size in gigabytes
        filesize_gb = os.path.getsize(os.path.join(experiment, fc_raw_file['name'])) / (1024 * 1024 * 1024)
        # check if file is over 2GB, if so then chunk the data in.
        if filesize_gb >= 2.0:
            fc_raw_reader = pd.read_json(os.path.join(experiment, fc_raw_file['name']), lines=True, chunksize=150)
            fc_raw_df_list = []
            # iterate over chunk dataframes and transform dataframes
            for cidx, chunck_df in enumerate(fc_raw_reader):
                print(f'chunk # {cidx + 1}')
                cols_to_explode = [col for col in chunck_df.columns if col != 'sample_id']
                chunck_df = explode(chunck_df, cols_to_explode)
                fc_raw_df_list.append(chunck_df)
            # return a concatenation of the chunked dataframes
            return pd.concat(fc_raw_df_list)
        else:
            fc_raw_data = pd.read_json(os.path.join(experiment, fc_raw_file['name']), orient='records', lines=True)
            cols_to_explode = [col for col in fc_raw_data.columns if col != 'sample_id']
            fc_raw_data = explode(fc_raw_data, cols_to_explode)
            return fc_raw_data
    else:
        return None


def get_data(experiment, record):
    fc_raw_file = get_record_file(record, file_type="fc_raw_events")
    if fc_raw_file:
        fc_raw_data = json.load(open(os.path.join(experiment, fc_raw_file['name'])))
        return json_to_pd(fc_raw_data)
    else:
        return None


def get_data_and_metadata_mem(experiment):
    record = get_record(experiment)
    data = get_data_mem(experiment, record)
    meta = get_meta(experiment, record)
    if data is not None and meta is not None:
        df = meta.merge(data, on="sample_id", how="inner")
        return df
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


