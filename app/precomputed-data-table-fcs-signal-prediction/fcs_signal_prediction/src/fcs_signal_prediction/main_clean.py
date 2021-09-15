#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from typing import Optional
from functools import reduce
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from fcs_signal_prediction.utils import data_utils as du
from pysd2cat.analysis import correctness


parser = argparse.ArgumentParser(description='Predict Signal Output for samples')
parser.add_argument('data_converge_path', type=str,
                    help='Path to data converge results')
parser.add_argument('experiment_id', type=str,
                    help='Experiment Identifier')
parser.add_argument('low_control', type=str,
                    help='Strain Name of Low Control')
parser.add_argument('high_control', type=str,
                    help='Strain Name of High Control')
parser.add_argument('hl_idx', type=int,
                    help='The High/Low control combo index')
parser.add_argument('out_dir', type=str,
                    help='The directory where results are saved to')
parser.add_argument('--id_col', type=str,
                    help='Sample id column name', default="sample_id")
parser.add_argument('--strain_col', type=str,
                    help='Strain column name', default="strain_name")


def perform_CV(x_data, y_data, plate_df):
    '''
    Builds a Random Forest Classifer and performs a 5-fold cross validation on the data. Returns the probabilities for each class for each data row.
        
        Parameters:
            x_data (pd.DataFrame): A dataframe containing the test data
            y_data (pd.Series): A series containing the class labels for data in x_data
            plate_df (pd.DataFrame): the original data form (x_data and y_data)

        Retruns:
            cv_plate_prob_pred_df (pd.DataFrame): The plate_df dataframe with the class probabilities for each data row appended as columns
    '''


    rf_model_cv = RandomForestClassifier(random_state=1, class_weight='balanced', n_estimators=361,  criterion='entropy', min_samples_leaf=13, n_jobs=-1)

    # Cross Validation
    print('Performing Cross Validation')
    cv_prob_list = list()
    cv_pred_list = list()

    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    x_ndarray = x_data.to_numpy()
    del x_data

    for i, (train_index, test_index) in enumerate(skf.split(x_ndarray, y_data)):
        print(f'Fold {i+1} of CV')
        X_train, X_test = x_ndarray[train_index], x_ndarray[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        scaler = StandardScaler().fit(X_train)

        train_X_norm = scaler.transform(X_train)
        if np.any(np.isnan(train_X_norm)):
            print(f'Number of NaNs in training data: {np.count_nonzero(np.isnan(train_X_norm))}')
            train_X_norm = np.nan_to_num(train_X_norm)

        val_X_norm = scaler.transform(X_test)            
        if np.any(np.isnan(val_X_norm)):
            print(f'Number of NaNs in test data: {np.count_nonzero(np.isnan(val_X_norm))}')
            val_X_norm = np.nan_to_num(val_X_norm)

        rf_model_cv.fit(train_X_norm, y_train)
        y_prob = rf_model_cv.predict_proba(val_X_norm)
        y_pred = rf_model_cv.predict(val_X_norm)

        prob_df = pd.DataFrame(y_prob, index=test_index)
        prob_df = prob_df.rename(columns={0: f'0_{i}', 1: f'1_{i}'})

        pred_df = pd.DataFrame(y_pred, index=test_index)
        pred_df = pred_df.rename(columns={0: f'0_{i}'})

        CV_test_AP = average_precision_score(y_pred, y_test)
        print(f'CV Fold {i+1} Test AP: {CV_test_AP}')

        x_train_pred = rf_model_cv.predict(train_X_norm)
        CV_train_AP = average_precision_score(x_train_pred, y_train)
        print(f'CV Fold {i+1} Train AP: {CV_train_AP}')

        cv_prob_list.append(prob_df)
        cv_pred_list.append(pred_df)

    df_prob_merge = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), cv_prob_list)
    df_prob_merge[df_prob_merge.columns.tolist()] = df_prob_merge[df_prob_merge.columns.tolist()].astype(float)
    pos_cols = [col for col in df_prob_merge.columns if col.startswith('1')]
    neg_cols = [col for col in df_prob_merge.columns if col.startswith('0')]
    pos_prob_df = df_prob_merge[pos_cols].copy()
    neg_prob_df = df_prob_merge[neg_cols].copy()
    pos_prob_df['pos_prob'] = pos_prob_df.max(skipna=True, axis=1)
    neg_prob_df['neg_prob'] = neg_prob_df.max(skipna=True, axis=1)
    final_prob_df = pd.DataFrame([pos_prob_df['pos_prob'], neg_prob_df['neg_prob']]).T

    df_pred_merge = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), cv_pred_list)
    df_pred_merge[df_pred_merge.columns.tolist()] = df_pred_merge[df_pred_merge.columns.tolist()].astype(float)
    final_pred_df = pd.DataFrame(df_pred_merge.max(skipna=True, axis=1).copy(), columns=['pred'])
    pred_prob_df = pd.merge(left=final_prob_df, right=final_pred_df, how='outer', left_index=True, right_index=True)

    print('Making final results from CV')
    # make final results
    reset_plate_df = plate_df.reset_index()

    cv_plate_prob_pred_df = pd.merge(left=reset_plate_df, right=pred_prob_df, how='outer', left_index=True, right_index=True)
    cv_plate_prob_pred_df = cv_plate_prob_pred_df.drop(labels='index', axis=1)

    return cv_plate_prob_pred_df


def train_test_model(x_data, y_data, model_type):
    '''
    Builds a Random Forest Classifer and performs a train/test split on the data. Prints out the training and testing average precision and returns the scalar and model.
        
        Parameters:
            x_data (pd.DataFrame): A dataframe containing the test data
            y_data (pd.Series): A series containing the class labels for data in x_data
            model_type (str): the type of model being made. 

        Retruns:
            scaler (sklearn.preprocessing.StandardScaler): the fitted scaler 
            rf_model (sklearn.ensemble.RandomForestClassifier): the trained random forest classifer
    '''

    print('Creating controls model')

    train_X, test_X, train_y, test_y = train_test_split(x_data, y_data, stratify=y_data, test_size=0.2, random_state=5)

    rf_model = RandomForestClassifier(random_state=1, class_weight='balanced', n_estimators=361,  criterion='entropy', min_samples_leaf=13, n_jobs=-1)

    scaler = StandardScaler().fit(train_X)  

    train_X_norm = scaler.transform(train_X)
    if np.any(np.isnan(train_X_norm)):
        print(f'Number of NaNs in training data: {np.count_nonzero(np.isnan(train_X_norm))}')
        train_X_norm = np.nan_to_num(train_X_norm)

    test_X_norm = scaler.transform(test_X)            
    if np.any(np.isnan(test_X_norm)):
        print(f'Number of NaNs in test data: {np.count_nonzero(np.isnan(test_X_norm))}')
        test_X_norm = np.nan_to_num(test_X_norm)

    rf_model.fit(train_X_norm, train_y)

    test_y_pred = rf_model.predict(test_X_norm)
    test_AP = average_precision_score(test_y_pred, test_y)
    print(f'{model_type} Model: Test AP: {test_AP}')

    train_y_pred = rf_model.predict(train_X_norm)
    train_AP = average_precision_score(train_y_pred, train_y)
    print(f'{model_type} Model: Train AP: {train_AP}')

    print('Confusion Matrix:')
    conf_mat = pd.DataFrame(confusion_matrix(test_y, test_y_pred, labels=[1, 0]), columns=['Positive', 'Negative'], index=['Positive', 'Negative'])
    print(conf_mat)

    return scaler, rf_model


def find_prob_events_balance(thres_range, min_events, class_df, col_name):
    '''
        Finds the highest probability threshold from a range of probabilities that maintains a minimum number of data points. Returns a single probability.

        Parameters:
             thres_range (list of floats): an ordered list of probabilities
             min_events (int): the minimum number of data points to keep
             class_df (pd.DataFrame): a dataframe containing the data
             col_name (): the column name containing the probabilities to threshold on

        Retruns:
            opt_thres (float): the probability from thres_range that meets the min_events criteria
    '''

    opt_thres = thres_range.min()
    for i, thres in enumerate(thres_range):
        if i > 0:
            clean_count = class_df[class_df[col_name] >= thres].shape[0]
            if clean_count >= min_events:
                opt_thres = thres
            elif clean_count < min_events:
                opt_thres = thres_range[i-1]
                break
    return round(opt_thres, 2)


def clean_controls(min_thres, max_thres, thres_step, min_events, positive_controls, negative_controls):
    '''
        Drops data points from each class (positive and negative controls) based on a probability threshold that drops the most data but maintains a given minimum of data points.
        Returns the "cleaned" data for each class.

        Parameters:
            min_thres (float): the minimum probability to use
            max_thres (float): the maximum probability to use
            thres_step (float): the step size to use when making the range of probabilities to use
            min_events (int): the minimum number of data points to keep
            positive_controls (pd.DataFrame): the dataframe containing data labeled as positives
            negative_controls (pd.DataFrame): the dataframe containing data labeled as negatives

        Retruns:
            clean_X (pd.DataFrame): the thresholded data
            clean_y (pd.Series): the thresholded data labels
            clean_controls_df (pd.DataFrame): the combined clean_X and clean_y
    '''

    print('Finding optimal thresholds')
    thres_range = np.arange(min_thres, max_thres+thres_step, thres_step)
    pos_thres = find_prob_events_balance(thres_range, min_events, positive_controls, 'pos_prob')
    neg_thres = find_prob_events_balance(thres_range, min_events, negative_controls, 'neg_prob')

    print('Building Clean Control Model')
    clean_pos_df = positive_controls[positive_controls['pos_prob'] >= pos_thres].iloc[:,:48].copy()
    clean_pos_df['class_label'] = 1

    clean_neg_df = negative_controls[negative_controls['neg_prob'] >= neg_thres].iloc[:,:48].copy()
    clean_neg_df['class_label'] = 0
    print()
    print(f'Positive Threshold: {pos_thres}')
    print(f'Original # of Positive Samples: {positive_controls.shape[0]}')
    print(f'# of Cleaned Positive Samples: {clean_pos_df.shape[0]}')
    print('Clean Positive Samples Description:')
    print(clean_pos_df.describe())
    print()
    print(f'Negative Threshold: {neg_thres}')
    print(f'Original # of Negative Samples: {negative_controls.shape[0]}')
    print(f'# of Cleaned Negative Samples: {clean_neg_df.shape[0]}')            
    print('Clean Negative Samples Description:')
    print(clean_neg_df.describe())
    print()
    clean_controls_df = pd.concat([clean_pos_df, clean_neg_df])
    print(f'Clean controls df shape: {clean_controls_df.shape}')
    print(f'Number of Positive (1) and Negative (0) classes:')
    print(clean_controls_df['class_label'].value_counts())

    clean_X = clean_controls_df.drop(columns=['class_label'])
    clean_y = clean_controls_df['class_label'].astype(int)

    return clean_X, clean_y, clean_controls_df


def pred_signal(scaler, rf_model, data_df, channels):
    '''
        Predicts on data using a trained model.

        Parameters:
            scaler (sklearn.preprocessing.StandardScaler): a scaler fitted to training data
            rf_model (sklearn.ensemble.RandomForestClassifier): the trained model
            data_df (pd.DataFrame): the data to predict on
            channels (list of strings): the feature names ordered as they were in the training data

        Retruns:
            pred_data_df (pd.DataFrame): the data_df with the predictions appended as columns for each data point
    '''

    # Predict on all data of a plate (controls + experimental)
    print(f'Predicting on all samples')
    plate_data_df = data_df.drop(columns=['sample_id']).copy()
    plate_data_df = plate_data_df[channels]
    # make sure columns are ordered the same

    print(f'Features: {plate_data_df.columns}')
    print(f'Shape of data to predict on: {plate_data_df.shape}')
    full_x_scaled = scaler.transform(plate_data_df)
    if np.any(np.isnan(full_x_scaled)):
        print(f'Number of NaNs in training data: {np.count_nonzero(np.isnan(full_x_scaled))}')
        full_x_scaled = np.nan_to_num(full_x_scaled)

    full_pred = rf_model.predict(full_x_scaled)
    full_pred_df = pd.DataFrame(full_pred, columns=['predicted_output'])
    pred_data_df = pd.concat([data_df, full_pred_df], axis=1)
    
    return pred_data_df


def save_log10_results(pred_df, class_pred_col, experiment_identifier, abrv_plate_id, hl_idx, model_type, out_dir):
    '''
        Separate the fcs data based on class predictions, bin the data and save the histograms.

        Parameters:
            pred_df (pd.DataFrame): the fcs data containing a column specifying the prediction made
            class_pred_col (str): the column containing the prediction
            experiment_identifier (str): the name of the experiment
            abrv_plate_id (srt): the abbreviated name of the plate
            hl_idx (int): the high/low combo used represented as an int
            model_type (str): the type of model used
            out_dir (str): the diretory to save the results to

        Retruns:
            log10_pred_fname (str): the name of the file the results were saved to
    '''

    print('Saving Log10 results')

    print('\tlog10 normalizing and binning data')
    print('\tCreating "pON" histograms')
    on_df = du.make_log10_df(pred_df, 1, class_pred_col)
    print('\tCreating "pOFF" histograms')
    off_df = du.make_log10_df(pred_df, 0, class_pred_col)
    print('\tCombining "pON" and "pOFF" histogram data')
    log10_pred_df = pd.concat([on_df, off_df])
    log10_pred_df.sort_values(by=['sample_id'], inplace=True)

    log10_pred_fname = f'pdt_{experiment_identifier}__{abrv_plate_id}_HL{int(hl_idx)+1}_{model_type}_fcs_signal_prediction__fc_raw_log10_stats.csv'
    log10_pred_df.to_csv(os.path.join(out_dir, log10_pred_fname), index=False)

    return log10_pred_fname


def save_pred_results(pred_df, meta_df, high_control, low_control, experiment_identifier, abrv_plate_id, hl_idx, id_col, model_type, out_dir):
    '''
        Computes statistics on the predictions and save the results.

        Parameters:
            pred_df (pd.DataFrame): the fcs data containing a column specifying the prediction made 
            meta_df (pd.DataFrame): the dataframe containing the metadata
            high_control (str): the name of the high control
            low_control  (str): the name of the low control
            experiment_identifier (str): the name of the experiment
            abrv_plate_id (srt): the abbreviated name of the plate
            hl_idx (int): the high/low combo used represented as an int
            id_col (str): the column containing the sample id
            model_type (str): the type of model used
            out_dir (str): the diretory to save the results to

        Retruns:
            results_fname (str): the name of the file the results were saved to
    '''

    ## Get the mean and std output signal for each sample 
    mean_prediction = pred_df.groupby([id_col]).agg({"predicted_output" : [np.mean, np.std]}).reset_index()
    mean_prediction.columns  = mean_prediction.columns.map('_'.join)
    mean_prediction = mean_prediction.rename(columns={id_col+"_": id_col})


    ## Get counts of each output signal prediction for each sample
    pred_df['predicted_output'] = pred_df['predicted_output'].astype(str)
    pred_df = pred_df.replace(['0', '1'], ['OFF', 'ON'])
    pred_df = pred_df.groupby(['sample_id', 'predicted_output'])['predicted_output'].agg('count').reset_index(name="p")
    pred_df = pred_df.pivot(index='sample_id', columns='predicted_output').reset_index()
    pred_df.columns = pred_df.columns.map(''.join)

    ## Attach the mean and std to the metadata
    meta_pred_stats = meta_df.merge(mean_prediction, on=id_col)
    result = meta_pred_stats.merge(pred_df, on='sample_id')

    result['high_control'] = high_control
    result['low_control'] = low_control
    results_fname = f'pdt_{experiment_identifier}__{abrv_plate_id}_HL{int(hl_idx)+1}_{model_type}_fcs_signal_prediction__fc_meta.csv'
    result.to_csv(os.path.join(out_dir, results_fname), index=False)

    return results_fname


def save_clean_controls(clean_data_df, meta_df, high_control, low_control, experiment_identifier, abrv_plate_id, hl_idx, out_dir):
    '''
        Computes statistics on the predictions and save the results.

        Parameters:
            clean_data_df (pd.DataFrame): the cleaned training fcs data containing a column specifying the prediction made 
            meta_df (pd.DataFrame): the dataframe containing the metadata
            high_control (str): the name of the high control
            low_control  (str): the name of the low control
            experiment_identifier (str): the name of the experiment
            abrv_plate_id (srt): the abbreviated name of the plate
            hl_idx (int): the high/low combo used represented as an int
            out_dir (str): the diretory to save the results to

        Retruns:
            clean_control_log10_fname (str): the name of the file the results were saved to
    '''

    high_sample_id = meta_df[meta_df['strain_name'] == high_control]['sample_id'].iloc[0]
    low_sample_id = meta_df[meta_df['strain_name'] == low_control]['sample_id'].iloc[0]

    pos_df = clean_data_df[clean_data_df['class_label'] == 1].copy()
    neg_df = clean_data_df[clean_data_df['class_label'] == 0].copy()
    
    pos_df['sample_id'] = high_sample_id
    neg_df['sample_id'] = low_sample_id

    clean_controls_sampid_df = pd.concat([pos_df, neg_df])
    clean_control_log10_fname = save_log10_results(clean_controls_sampid_df, 'class_label', experiment_identifier, abrv_plate_id, hl_idx, 'cleanControls', out_dir)

    return clean_control_log10_fname


def main(data_converge_path: str, 
         experiment_identifier: str,
         low_control : str,
         high_control : str,
         hl_idx: str,
         out_dir: str,
         id_col : Optional[str]="sample_id",
         strain_col : Optional[str]='strain_name'):
    """
    This analysis has two parts after retrieving the raw fcs cytometry data
    1)  train model on all positive and negative control data,
        predict the output signal for each event,
        aggregate by sample,
        return metadata with columns for mean and std dev. of the predicted signal for each sample.
    2)  clean the positive and negative control data based on probabilities,
        train model on cleaned positive and negative control data,
        predict the output signal for each event,
        aggregate by sample, 
        return metadata with columns for mean and std dev. of the predicted signal for each sample.
    """
    
    meta = du.get_meta(data_converge_path, du.get_record(data_converge_path))
    plate_samples_dict = du.make_plate_samples_dict(meta)
    
    ## Run FSP on each plate separately.
    results_fname_list = list()

    for plate_id in plate_samples_dict.keys():

        abrv_plate_id = plate_id.split('.')[-1]

        print()
        ## Get the data
        print(f'_____________ Plate Selected: {plate_id} _____________')
        data_df = du.grab_plate_samples_df(plate_id, plate_samples_dict, data_converge_path)
        print(f'Total events collected from plate {plate_id}: {data_df.shape[0]}')
        meta_data_df = data_df.merge(meta[[id_col, strain_col]])
        
        ## Get the channels in the data
        channels = list(meta_data_df.columns)
        channels.remove(id_col)
        channels.remove(strain_col)
        
        plate_df = correctness.get_classifier_dataframe(meta_data_df, data_columns=channels, strain_col=strain_col, high_control=high_control, low_control=low_control)
        del meta_data_df
        print('Controls dataframe created')

        full_control_X = plate_df.drop(columns=['class_label'])
        full_control_y = plate_df['class_label'].astype(int)

        # full control mode
        full_control_scaler, full_control_rf_model = train_test_model(full_control_X, full_control_y, 'Full Controls')

        # predict on all data using full control model
        full_model_pred_data_df = pred_signal(full_control_scaler, full_control_rf_model, data_df, channels)

        # save results
        fullmodel_log10_fname = save_log10_results(full_model_pred_data_df, 'predicted_output', experiment_identifier, abrv_plate_id, hl_idx, 'fullModel', out_dir)
        results_fname_list.append(fullmodel_log10_fname)
        fullmodel_pred_fname = save_pred_results(full_model_pred_data_df, meta, high_control, low_control, experiment_identifier, abrv_plate_id, hl_idx, id_col, 'fullModel', out_dir)
        results_fname_list.append(fullmodel_pred_fname)
        del full_model_pred_data_df

        # perform CV and get probabilities
        cv_plate_prob_pred_df = perform_CV(full_control_X, full_control_y, plate_df)
        del full_control_X
        del full_control_y

        # Build new model on "cleaned" controls
        pos_df = cv_plate_prob_pred_df[cv_plate_prob_pred_df['class_label'] == 1].copy()
        neg_df = cv_plate_prob_pred_df[cv_plate_prob_pred_df['class_label'] == 0].copy()
        del cv_plate_prob_pred_df     
        print('Original Positive Samples "class_label" Description:')
        print(pos_df['class_label'].describe())
        print('Original Negative Samples "class_label" Description:')
        print(neg_df['class_label'].describe())
        print()
        print()

        # clean controls
        clean_x, clean_y, clean_controls_df = clean_controls(0.5, 0.95, 0.01, 10000, pos_df, neg_df)
        del pos_df
        del neg_df

        cleancontrols_log10_fname = save_clean_controls(clean_controls_df, meta, high_control, low_control, experiment_identifier, abrv_plate_id, hl_idx, out_dir)
        results_fname_list.append(cleancontrols_log10_fname)

        # Train/Test Clean Controls Model
        clean_scaler, clean_controls_rf_model = train_test_model(clean_x, clean_y, 'Clean Controls')
        del clean_x
        del clean_y
        # Predict using clean controls model
        clean_model_pred_df = pred_signal(clean_scaler, clean_controls_rf_model, data_df, channels)

        # save results
        cleanmodel_log10_fname = save_log10_results(clean_model_pred_df, 'predicted_output', experiment_identifier, abrv_plate_id, hl_idx, 'cleanModel', out_dir)
        results_fname_list.append(cleanmodel_log10_fname)
        cleanmodel_pred_fname = save_pred_results(clean_model_pred_df, meta, high_control, low_control, experiment_identifier, abrv_plate_id, hl_idx, id_col, 'cleanModel', out_dir)
        results_fname_list.append(cleanmodel_pred_fname)
        del clean_model_pred_df
        print()
        print()

    return results_fname_list

if __name__ == '__main__':
    args = parser.parse_args()

    data_converge_path = args.data_converge_path
    experiment_identifier = args.experiment_id
    low_control = args.low_control
    high_control = args.high_control
    hl_idx = args.hl_idx
    out_dir = args.out_dir
    id_col = args.id_col
    strain_col = args.strain_col
    
    main(data_converge_path, 
         experiment_identifier,
         low_control,
         high_control,
         hl_idx,
         out_dir,
         id_col=id_col,
         strain_col=strain_col)