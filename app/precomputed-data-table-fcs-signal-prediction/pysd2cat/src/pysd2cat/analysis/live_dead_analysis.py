import sys
import os
import json
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import logging
l = logging.getLogger(__file__)
l.setLevel(logging.INFO)


from pysd2cat.data import pipeline
from pysd2cat.analysis import threshold as thold
from pysd2cat.analysis import live_dead_classifier as ldc
from pysd2cat.analysis.Names import Names
from pysd2cat.plot import plot


def write_live_dead_column(data_file, strain_column_name, live_strain_name, dead_strain_name, output_col, fcs_columns, out_dir, overwrite=False, description=None, random_state=5, dry_run=False, feature_importance=False):
    df = None
    try:
        #data_dir = "/".join(data_file.split('/')[0:-1])
        #out_path = os.path.join(data_dir, 'accuracy')
        #out_file = os.path.join(out_path, data_file.split('/')[-1])
        df = pd.read_csv(data_file, dtype={'od': float, 'input' : object, 'output' : object}, index_col=0)
        #print(df.head(5))
        if not dry_run and not overwrite and output_col in df.columns:
            print("Already done: " + data_file)
        elif not dry_run:
            strains = df[strain_column_name].unique()
            #print("strains: {}".format(type(strains)))
            #print(strains)
            if live_strain_name in strains and dead_strain_name in strains:

                print("Computing live/dead for: " + data_file)
                df = add_live_dead_test_harness(df, strain_column_name, live_strain_name, dead_strain_name, out_dir=out_dir, output_col= output_col, fcs_columns=fcs_columns, description=description, random_state=random_state, feature_importance=feature_importance)
                print("Writing live/dead for: " + data_file)
                df.to_csv(data_file)
        else:
            # dry run, just build classifier
            df = add_live_dead_test_harness(df, strain_column_name, live_strain_name, dead_strain_name, out_dir=out_dir, output_col= output_col, fcs_columns=fcs_columns, description=description, random_state=random_state, dry_run=dry_run, feature_importance=feature_importance)
    except Exception as e:
        print("File failed: " + data_file + " with: " + str(e))
        
    #return df

def write_live_dead_column_job(args, kwargs):
    write_live_dead_column(*args, **kwargs)



def write_live_dead_columns(data, strain_column_name, live_strain_name, dead_strain_name, fcs_columns, output_col='live', num_cpu=None, overwrite=False, description=None, out_dir='.', random_state=5, dry_run=False):
    import multiprocessing
    if num_cpu:
        pool = multiprocessing.Pool(num_cpu)
    else:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    tasks = []
    for d in data:
        run_id=d.split("/")[-1].split('.')[0]
        tasks.append(((d, strain_column_name, live_strain_name, dead_strain_name, output_col, fcs_columns, out_dir),
                     {'overwrite': overwrite, 'description' : run_id + "_" + str(random_state) + "_" + description, 'random_state' : random_state, 'dry_run' : dry_run}))
    results = [pool.apply_async(write_live_dead_column_job, t) for t in tasks]

    for result in results:
        data_list = result.get()

def strain_to_class(x,
                    strain_column_name,
                    live_strain_name,
                    dead_strain_name):
    """
    Boolean class labels for live/dead classifier
    """
    if x[strain_column_name] == live_strain_name:
        return 1
    elif x[strain_column_name] == dead_strain_name:
        return 0
    else:
        return None

def get_classifier_dataframe(df,
                             strain_column_name,
                             live_strain_name,
                             dead_strain_name,
                             fcs_columns = None):
    """
    Get the classifier data corresponding to controls.
    """
    df_columns = df.columns.tolist()
    l.info("data_columns: {}".format(df_columns))
    live_dead_df = df.loc[(df[strain_column_name].str.contains(live_strain_name)) | (df[strain_column_name].str.contains(dead_strain_name))]
    l.info("Writing class labels ...")
    live_dead_df.loc[:,strain_column_name] = live_dead_df.apply(lambda x : strain_to_class(x, strain_column_name, live_strain_name, dead_strain_name), axis=1)
    l.info(live_dead_df)
    live_dead_df = live_dead_df.rename(index=str, columns={strain_column_name: "class_label"})
    if fcs_columns:
        data_columns = fcs_columns
    else:
        data_columns = list(set(df_columns) - {'sample_id', strain_column_name, 'file_id'})
    l.info("data_columns: %s", str(data_columns))
    live_dead_df = live_dead_df[data_columns + ['class_label']]
    return live_dead_df

def get_multi_classifier_dataframe(df,
                             strain_column_name,
                             classes,
                             fcs_columns = None):
    """
    Get the classifier data corresponding to controls.
    """
    df_columns = df.columns.tolist()
    print("data_columns: {}".format(df_columns))

    classes.sort()
    live_dead_df = pd.DataFrame()
    for i, class_value in enumerate(classes):
        class_df = df.loc[df[strain_column_name] == class_value]
        class_df.loc[:, strain_column_name] = i
        live_dead_df = live_dead_df.append(class_df)

    live_dead_df = live_dead_df.rename(index=str, columns={strain_column_name: "class_label"})
    if fcs_columns:
        data_columns = fcs_columns
    else:
        data_columns = list(set(df_columns) - {'sample_id', strain_column_name, 'file_id'})
    print(data_columns)
    live_dead_df = live_dead_df[data_columns + ['class_label']]
    return live_dead_df

def add_live_dead_test_harness(df,
                               strain_column_name,
                               live_strain_name,
                               dead_strain_name,
                               classifier_df=None,
                               output_col='live',
                               fcs_columns = None,   
                               description=None,
                               out_dir='.',
                               random_state=5,
                               dry_run=False,
                               feature_importance=False):
    """
    Same as add_live_dead(), but use test-harness.
    """
 

    ## Build the training/test input
    ## Use classifier_df to build classifier, but predict in df
    if classifier_df is None:
        l.info("Extracting controls from data to predict ...")
        classifier_df = df
    l.info("Getting Data to use for Train and Test ...")
    c_df = get_classifier_dataframe(classifier_df,
                                    strain_column_name,
                                    live_strain_name,
                                    dead_strain_name,
                                    fcs_columns = fcs_columns)
    l.info("Have Train and Test Data ...")
    #print(c_df.head())
    c_df.loc[:, 'index'] = c_df.index
    #c_df.loc[:, 'id'] = df['id']
    df.loc[:, 'index'] = df.index

    l.info("Determining data_columns ...")
    if fcs_columns:
        data_columns = fcs_columns
        #ouput_columns = list(df.columns) + [output_col]
    else:
        data_columns = list(set(df.columns) - {'sample_id', strain_column_name, 'file_id'})
       # ouput_columns = ['sample_id'] + data_columns + [output_col]

    l.info("Using data_columns: %s", str(data_columns))
    if not description:
        experiment = df.lab_id.unique()[0]
        description = experiment+"_live"

    ## Build the classifier
    ## Predict label for unseen data
    l.info("Building the model ...")
    pred_df = ldc.build_model_pd(c_df, 
                                 data_df = df, 
                                 input_cols=data_columns,
                                 index_cols=['index'],
                                 output_location=out_dir,
                                 description=description,
                                 random_state=random_state,
                                 dry_run=dry_run,
                                 feature_importance=feature_importance
                                )
    #print("output_col: " + output_col)
    if not dry_run:
        #print(pred_df.head())
        if output_col in df.columns:
            #print("Dropping column " + output_col)
            df=df.drop([output_col], axis=1)
        #df=df.reset_index()
        live_col = pred_df.rename(columns={'class_label_predictions': output_col})[[output_col]]
        #live_col.index = live_col.index.astype(str)
        #print(live_col.dtypes)

        #print(live_col.sort_index().head())
        #print(df.dtypes)
        df = df.join(live_col, how='left')
        #print(df.sort_index().head())
    
    return df

def add_live_dead_multi_test_harness(df,
                               strain_column_name,
                               classes,
                               classifier_df=None,
                               output_col='live',
                               fcs_columns = None,   
                               description=None,
                               out_dir='.',
                               random_state=5,
                               dry_run=False,
                               feature_importance=False):
    """
    Same as add_live_dead(), but use test-harness.
    """
 

    ## Build the training/test input
    ## Use classifier_df to build classifier, but predict in df
    if classifier_df is None:
        l.info("Extracting controls from data to predict ...")
        classifier_df = df
    l.info("Getting Data to use for Train and Test ...")
    c_df = get_multi_classifier_dataframe(classifier_df,
                                    strain_column_name,
                                    classes,
                                    fcs_columns = fcs_columns)

    #print(c_df.head())
    c_df.loc[:, 'index'] = c_df.index
    #c_df.loc[:, 'id'] = df['id']
    df.loc[:, 'index'] = df.index

    if fcs_columns:
        data_columns = fcs_columns
        #ouput_columns = list(df.columns) + [output_col]
    else:
        data_columns = list(set(df.columns) - {'sample_id', strain_column_name, 'file_id'})
       # ouput_columns = ['sample_id'] + data_columns + [output_col]

    l.info("Using data_columns: %s", str(data_columns))
    if not description:
        experiment = df.lab_id.unique()[0]
        description = experiment+"_live"

    ## Build the classifier
    ## Predict label for unseen data
    l.info("Building the model ...")
    pred_df = ldc.build_model_pd(c_df, 
                                 data_df = df, 
                                 input_cols=data_columns,
                                 index_cols=['index'],
                                 output_location=out_dir,
                                 description=description,
                                 random_state=random_state,
                                 dry_run=dry_run,
                                 feature_importance=feature_importance
                                )
    #print("output_col: " + output_col)
    if not dry_run:
        #print(pred_df.head())
        if output_col in df.columns:
            #print("Dropping column " + output_col)
            df=df.drop([output_col], axis=1)
        #df=df.reset_index()
        live_col = pred_df.rename(columns={'class_label_predictions': output_col})[[output_col]]
        #live_col.index = live_col.index.astype(str)
        #print(live_col.dtypes)

        #print(live_col.sort_index().head())
        #print(df.dtypes)
        df = df.join(live_col, how='left')
        #print(df.sort_index().head())
    
    return df

def get_fcs_columns():
    return ['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H',
            'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W']

def add_live_dead(df,
                  strain_column_name,
                  live_strain_name,
                  dead_strain_name,
                  fcs_columns = None):
    """
    Take an input_df corresponding to one plate.
    Build a live dead classifier from controls.
    Apply classifier to each event to create 'live' column
    """

    ## Build the training/test input
    c_df = get_classifier_dataframe(df,
                                    strain_column_name,
                                    live_strain_name,
                                    dead_strain_name,
                                    fcs_columns = fcs_columns)
    print(c_df)
    ## Build the classifier
    (model, mean_absolute_error, test_X, test_y, scaler) = ldc.build_model(c_df)
    print(df.columns)
    ## Predict label for unseen data
    if fcs_columns:
        data_columns = fcs_columns
        ouput_columns = list(df.columns) + ['live']
    else:
        data_columns = list(set(df.columns) - {'sample_id', strain_column_name, 'file_id'})
        ouput_columns = ['sample_id'] + data_columns + ['live']
    print(data_columns)
    pred_df = df[data_columns]
    print(pred_df)
    pred_df = ldc.predict_live_dead(pred_df, model, scaler)
    df.loc[:,'live'] = pred_df['class_label'].astype(int)
    result_df = df[ouput_columns]
    print(result_df)
    return result_df


def compute_sytox_threshold_accuracy(live_dead_df, thresholds=[2000], sytox_channel='RL1-A'):
    value_df = live_dead_df[[sytox_channel, 'class_label']].rename(index=str, columns={sytox_channel: "value", 'class_label': 'output'})
    value_df['output'] = value_df['output'].apply(lambda x: math.fabs(x-1))
    value_df = value_df.sample(frac=0.1, replace=True)
    plot_df = thold.do_threshold_analysis(value_df, thresholds)

    return plot_df

def compute_model_predictions(model, scaler, circuits, inputs, ods, media, fraction=0.06, data_dir='.'):
    predictions = pd.DataFrame()
    for circuit in circuits:
        for input in inputs:
            for od in ods:
                for m in media:
                    m_df = pipeline.get_strain_dataframe_for_classifier(circuit, input, od=od, media=m, data_dir=data_dir, fraction=fraction)
                    if m_df is not None:
                        #print(circuit + " " + input + " " + str(od))
                        pred_df = ldc.predict_live_dead(m_df.drop(columns=['output']), model, scaler)
                        m_df['class_label'] = pred_df['class_label']
                        m_df['circuit'] = circuit
                        m_df['input'] = input
                        m_df['media'] = m
                        m_df['od'] = od
                        predictions = predictions.append(m_df, ignore_index=True)
    return predictions

def compare_accuracy_of_gating(model, scaler, circuits, inputs, ods, media, fraction=0.06, data_dir='.', channel='BL1-A', thresholds=[10000]):

    plot_df = pd.DataFrame()
    for circuit in circuits:
        for input in inputs:
            for od in ods:
                for m in media:
                    m_df = pipeline.get_strain_dataframe_for_classifier(circuit, input, od=od, media=m, data_dir=data_dir, fraction=fraction)
                    if m_df is not None:
                        #print(m_df.head())
                        pred_df = ldc.predict_live_dead(m_df.drop(columns=['output']), model, scaler)
                        m_df['class_label'] = pred_df['class_label']
                        gated_df = m_df.loc[m_df['class_label'] == 1] # live cells
                        num_gated = len(m_df.index) - len(gated_df.index)
    
                        value_df = m_df[[channel, 'output']].rename(index=str, columns={channel: "value"})
                        gated_value_df = gated_df[[channel, 'output']].rename(index=str, columns={channel: "value"})
    
                        thold_df = thold.do_threshold_analysis(value_df, thresholds)
                        gated_thold_df = thold.do_threshold_analysis(gated_value_df, thresholds)
    
                        #print(thold_df)
                        thold_df['circuit'] = circuit
                        thold_df['input'] = input
                        thold_df['media'] = m
                        thold_df['od'] = od
                        thold_df['num_gated'] = num_gated
    
                        thold_df['gated_probability_correct'] = gated_thold_df['probability_correct']
                        thold_df['gated_standard_error_correct'] = gated_thold_df['standard_error_correct']
                        thold_df['gated_count'] = gated_thold_df['count']
    
                        #print(thold_df)
                        plot_df = plot_df.append(thold_df, ignore_index=True)
    return plot_df

def plot_strain_live_dead_predictions(predictions, circuits, inputs, ods, media, filename='live_dead_predictions.pdf'):
    with PdfPages(filename) as pdfpages:
        for circuit in circuits:
            for od in ods:
                for m in media:
                    title=circuit+"_"+str(od)
                    #print(title)
                    pdf = predictions.loc[(predictions['circuit'] == circuit) & (predictions['od'] == od) & (predictions['media'] == m) ]
                    plot.plot_live_dead_experiment(pdf, inputs, title, plot_dir='plot', pdf=pdfpages)



def main():
    ## Where data files live
    ##HPC
    data_dir = '/work/projects/SD2E-Community/prod/data/uploads/'

    ##Jupyter Hub
    #data_dir = '/home/jupyter/sd2e-community/'


    ods=[0.0003, 0.00015, 7.5e-05]
    media=['SC Media']
    circuits=['XOR', 'XNOR', 'OR', 'NOR', 'NAND', 'AND']
    inputs=['00', '01', '10', '11']
    data_fraction=0.06

    print("Building Live/Dead Control Dataframe...")
    live_dead_df = pipeline.get_dataframe_for_live_dead_classifier(data_dir)
    print(live_dead_df.columns.tolist())
    print(live_dead_df.head(5))

    print("Training Live/Dead Classifier...")
    (model, mean_absolute_error, test_X, test_y, scaler) = ldc.build_model(live_dead_df)
    print("MAE = " + str(mean_absolute_error))

    print("Plotting Classifier Predictions on Test Set by channel...")
    plot.plot_live_dead_control_predictions_by_channel(test_X, test_y,
                                                     filename='live_dead_control_model_channels.png')

    print("Computing Sytox Threshold Accuracy...")
    thresholds=[1000, 2000, 2500, 5000]
    threshold_analysis = compute_sytox_threshold_accuracy(live_dead_df, thresholds=thresholds)

    print("Plotting Sytox Threshold Accuracy...")
    plot.plot_live_dead_threshold(threshold_analysis, 'Live/Dead Sytox Threshold',
                                  filename='live_dead_control_threshold_accuracy.png')


    print("Computing mean number of live cells per sample (comparing threshold to classifier)...")
    mean_live = ldc.compute_mean_live(model,
                                      scaler,
                                      data_dir,
                                      threshold=2000,
                                      ods=ods,
                                      media=media,
                                      circuits=circuits,
                                      inputs=inputs,
                                      fraction=data_fraction)
    mean_live.to_csv('mean_live.csv')


    print("Computing Live/Dead Predictions for strains...")
    predictions = compute_model_predictions(model, scaler, circuits, inputs, ods, media, fraction=data_fraction, data_dir=data_dir)

    print("Plotting Live/Dead Predictions for strains...")
    plot_strain_live_dead_predictions(predictions, circuits, inputs, ods, media)

    print("Performing Comparison of Accuracy Before and After gating with Live/Dead classifier")
    gating_comparison = compare_accuracy_of_gating(model, scaler, circuits, inputs, ods, media, fraction=data_fraction, data_dir=data_dir,thresholds=[10000])
    gating_comparison.to_csv('gating_comparison.csv')

if __name__ == '__main__':
    main()
