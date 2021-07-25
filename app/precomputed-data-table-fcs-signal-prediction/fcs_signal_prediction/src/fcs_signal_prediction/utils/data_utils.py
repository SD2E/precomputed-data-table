import json
import os
import numpy as np
import pandas as pd
from collections import OrderedDict


round_log_decimal = 5

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
                # print(f'chunk # {cidx+1}')
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


def get_plate_ids(data_converge_path, experiment_ref):

    plate_ids_list = list()

    exp_ids_list = get_record(data_converge_path)['status_upstream']
    for ids in exp_ids_list:
        if experiment_ref not in ids['experiment_id'] and 'F' in ids['data_types']:
            plate_ids_list.append(ids['experiment_id'])
    return plate_ids_list


def make_plate_samples_dict(meta_df):
    ps_dict = dict()
    exp_ids = set(meta_df['experiment_id'])
    for id in exp_ids:
        ps_dict[id] = meta_df[meta_df['experiment_id'] == id]['sample_id'].tolist()
    return ps_dict


def match_exp_ids(ps_dict, plate_id_list):
    for id in plate_id_list:
        if id in ps_dict.keys():
            print(f'{id} in record and FC metadata.\n\tNumber of samples: {len(ps_dict[id])}')


def grab_plate_samples_df(plate_id, ps_dict, data_converge_path):
    fc_raw_file = get_record_file(get_record(data_converge_path), file_type="fc_raw_events")
    plate_samples = []
    with open(os.path.join(data_converge_path, fc_raw_file['name'])) as f:
        for line in f:
                sample_line = json.loads(line)
                if sample_line['sample_id'] in ps_dict[plate_id]:
                    plate_samples.append(sample_line)
    fc_raw_df = pd.DataFrame(plate_samples)
    cols_to_explode = [col for col in fc_raw_df.columns if col != 'sample_id']
    return explode(fc_raw_df, cols_to_explode)



def log_transform(original_sample, channels):
    """
    This function implements a log10 transformation on the data.
    It is preferable to the FCT transforms because they include
    some kind of normalization that I don't understand.
    By: Devin Strickland
    """
    # Copy the original sample
    new_data = original_sample.copy()
    # log10 transformation of each specified channel
    for c in channels:
        new_data[c] = np.log10(new_data[c])
    new_data = new_data.replace([np.inf, -np.inf], np.nan)
    return new_data


def bin_data(pts, bin_endpoints):
    """
    Function to sort points into bins. All but the last bin is half-open (right side of interval is open). In other
    words, if bin_endpoints is: `[1, 2, 3, 4]` then the first bin is [1, 2) (including 1, but excluding 2) and the second
    [2, 3). The last bin, however, is [3, 4], which includes 4.

    Note: NAs are counted and then dropped prior to computing the counts for the other bins. This count is added to
    the beginning of the bins. Additionally we also add -infinity as the lower bound and infinity as the upper bound for
    our computed bins.

    :param pts: Input data
    :param bin_endpoints: Monotonically increasing array of bin edges
    :return: the counts for each bin
    """

    # count and drop na values
    na_bin = np.count_nonzero(np.isnan(pts))
    pts_wo_na = pts[np.logical_not(np.isnan(pts))]

    # get bin from (-infinity, min(bin_endpoints))
    lower_bin = (pts_wo_na < np.min(bin_endpoints)).sum()

    # get bin from (max(bin_endpoints), infinity)
    upper_bin = (pts_wo_na > np.max(bin_endpoints)).sum()

    # get counts for each of the bins
    all_bins, _ = np.histogram(pts_wo_na, bin_endpoints)

    # add na bin, lower bin, and upper bin to the bins for the rest of the data
    all_bins = np.append(na_bin, all_bins)

    return all_bins


def make_log10_df(fcs_data, prediction):

    pred_fcs_data = fcs_data[fcs_data['predicted_output'] == prediction]
    fcs_sample_list = list(set(pred_fcs_data['sample_id'].tolist()))

    bin_col_names = [0.1 * x for x in range(0, 71)]
    bin_col_names = ['bin(log10)_' + str(round(bin + 0.05, 2)) for bin in bin_col_names]
    bin_col_names = ['bin(log10)_NAN'] + bin_col_names

    fcs_record_list = []

    for sample in fcs_sample_list:

        try:

            fcs_data = pred_fcs_data[pred_fcs_data['sample_id'] == sample]
            print(f'sample_id: {sample} | Prediction: {prediction} | # of events: {fcs_data.shape[0]}')
            
            # get channel names, ignoring the sample_id and predicted_output columns
            ignore_channels = ['sample_id', 'predicted_output']
            channel_names = [ch for ch in fcs_data.columns if ch not in ignore_channels]

            # loop over channel names to perform log10 transformations
            for channel in channel_names:

                single_fcs_dict = OrderedDict()
                single_fcs_dict['sample_id'] = sample
                single_fcs_dict['channel'] = channel
                single_fcs_dict['predicted_output'] = prediction
                # log10 transform data
                log_data_fcs_df = log_transform(fcs_data, [channel])
                # get states: mean, std
                single_fcs_dict['mean_log10'] = round(log_data_fcs_df[channel].mean(), round_log_decimal)
                single_fcs_dict['std_log10'] = round(log_data_fcs_df[channel].std(), round_log_decimal)

                # bin data and add histogram data to dataframe
                binned_log_data = bin_data(log_data_fcs_df[channel], [0.1 * x for x in range(0, 71)])
                for bin_col, single_bin_data in zip(bin_col_names, binned_log_data):
                    single_fcs_dict['{}'.format(bin_col)] = single_bin_data

                fcs_record_list.append(single_fcs_dict)
        except:
            print("PROBLEM SAMPLE: ", sample)

    # Convert the rowWells dictionary created above into a dataframe
    fcs_row_wells_df = pd.DataFrame(fcs_record_list)
    fcs_row_wells_df.sort_values(by=['sample_id'])

    return fcs_row_wells_df