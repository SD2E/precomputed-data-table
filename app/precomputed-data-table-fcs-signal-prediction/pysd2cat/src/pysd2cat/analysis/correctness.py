from pysd2cat.analysis.Names import Names
from pysd2cat.analysis import live_dead_classifier as ldc
import numpy as np
from pysd2cat.analysis.threshold import compute_correctness
import pandas as pd
import os

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


def strain_to_class(x,  strain_col=Names.STRAIN, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    if x[strain_col] == low_control:
        return 0
    elif x[strain_col] == high_control:
        return 1
    else:
        return None


def get_classifier_dataframe(df, data_columns = ['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W'],
                                 strain_col=Names.STRAIN,
                             high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    """
    Get the classifier data corresponding to controls.
    """
    l.info("get_classifier_dataframe(): strain_col = " + strain_col)
    low_df = df.loc[df[strain_col] == low_control]
    high_df = df.loc[df[strain_col] == high_control]
    low_df.loc[:,strain_col] = low_df.apply(lambda x : strain_to_class(x,  high_control=high_control, low_control=low_control, strain_col=strain_col), axis=1)
    high_df.loc[:,strain_col] = high_df.apply(lambda x : strain_to_class(x,  high_control=high_control, low_control=low_control, strain_col=strain_col), axis=1)
    low_high_df = low_df.append(high_df)

    #print(live_dead_df)
    low_high_df = low_high_df.rename(index=str, columns={strain_col: "class_label"})
    low_high_df = low_high_df[data_columns + ['class_label']]
    return low_high_df


def compute_predicted_output(df, 
                             training_df=None,
                             data_columns = ['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 
                                             'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 
                                             'BL1_W', 'RL1_W'], 
                             out_dir='.',
                             strain_col=Names.STRAIN,
                             high_control=Names.NOR_00_CONTROL, 
                             low_control=Names.WT_LIVE_CONTROL,
                             id_col='id',
                             use_harness=False,
                             description=None):
    ## Build the training/test input
    if training_df is None:
        c_df = get_classifier_dataframe(df, data_columns = data_columns, strain_col=strain_col, high_control=high_control, low_control=low_control)
    else:
        c_df = training_df
    

    if use_harness:
        c_df.loc[:, 'output'] = df['output']
        c_df.loc[:, id_col] = df[id_col]
        c_df.loc[:, 'index'] = c_df.index
        df.loc[:,'index'] = df.index
        pred_df = ldc.build_model_pd(c_df,
                                     data_df = df,
                                     index_cols=['index', 'output', id_col],
                                     input_cols=data_columns,
                                     output_cols=['class_label'],
                                     output_location=out_dir,
                                     description=description)
        #result_df.loc[:,'predicted_output'] = pred_df['class_label_predictions'].astype(int)        
        result_df = pred_df.rename(columns={'class_label_predictions' : 'predicted_output'})
    else:
        ## Build the classifier
        (model, mean_absolute_error, test_X, test_y, scaler) = ldc.build_model(c_df)
        #print("MAE: " + str(mean_absolute_error))
        ## Predict label for unseen data
        pred_df = df[data_columns]
        pred_df = ldc.predict_live_dead(pred_df, model, scaler)
        if 'output' in df.columns:
            result_df = df[['output', id_col]]
        else:
            result_df = df[[id_col]]
        result_df.loc[:,'predicted_output'] = pred_df['class_label'].astype(int)
    
    return result_df

   
def compute_correctness_classifier(df,
                             training_df=None,
                             out_dir = '.',
                             mean_output_label='probability_correct',
                             std_output_label='std_probability_correct',
                             mean_correct_high_name='mean_correct_high_classifier',
                             std_correct_high_name='std_correct_high_classifier',
                             mean_correct_low_name='mean_correct_low_classifier',
                             std_correct_low_name='std_correct_low_classifier',
                             high_control=Names.NOR_00_CONTROL,
                             low_control=Names.WT_LIVE_CONTROL,
                             strain_col=Names.STRAIN,
                             description=None,
                             add_predictions = False,
                             use_harness = False,
                             id_col='id',
                             channels = ['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W'],
                             logger=l):
    
    df.loc[:,'output'] = pd.to_numeric(df['output'])
    logger.debug("Shape at start: " + str(df.shape) + " strain_col = " + strain_col)
    result_df = compute_predicted_output(df, 
                                         training_df=training_df,
                                         out_dir=out_dir,
                                         strain_col=strain_col,
                                         high_control=high_control, 
                                         low_control=low_control, 
                                         use_harness=use_harness,
                                         data_columns=channels,
                                         id_col=id_col,
                                         description=description)
    logger.debug("Shape of result: " + str(result_df.shape) + ", columns= " + str(result_df.columns))
    result_df.loc[:, 'predicted_correct'] = result_df.apply(lambda x : None if np.isnan(x['output']) or np.isnan(x['predicted_output']) else 1.0 - np.abs(x['output'] - x['predicted_output']), axis=1)
    result_df.loc[:, 'predicted_high'] = result_df.apply(lambda x : None if np.isnan(x['output']) or np.isnan(x['predicted_output']) else 1.0 - np.abs(1- x['predicted_output']), axis=1)
    result_df.loc[:, 'predicted_low'] = result_df.apply(lambda x : None if np.isnan(x['output']) or np.isnan(x['predicted_output']) else 1.0 - np.abs(x['predicted_output']), axis=1)



    logger.debug(result_df)
    if add_predictions:
        df.loc[:, 'predicted_output'] = result_df['predicted_output']
        df.loc[:, 'predicted_correct'] = result_df['predicted_correct']
        df.loc[:, 'predicted_high'] = result_df['predicted_high']
        df.loc[:, 'predicted_low'] = result_df['predicted_low']




    #acc_df = result_df.groupby(['id'])['predicted_correct'].agg([np.nanmean, np.nanstd]).reset_index()
    
    def nan_agg(x):
        res = {}
        x = x.dropna()
        if len(x) > 0:
#            print(x['predicted_correct'].value_counts())
            res['mean'] = x['predicted_correct'].mean()
            res['std'] = x['predicted_correct'].std()
            res['mean_high'] = x['predicted_high'].mean()
            res['std_high'] = x['predicted_high'].std()
            res['mean_low'] = x['predicted_low'].mean()
            res['std_low'] = x['predicted_low'].std()

        else:
            res['mean'] = None
            res['std'] = None
            res['mean_high'] = None
            res['std_high'] = None
            res['mean_low'] = None
            res['std_low'] = None


        #print(res['count'])
        return pd.Series(res, index=['mean', 'std', 'mean_high', 'std_high', 'mean_low', 'std_low'])

    groups = result_df.groupby([id_col])
    acc_df = groups.apply(nan_agg).reset_index() 

    logger.debug("Shape of acc: " + str(acc_df.shape))
    #print(acc_df)
    acc_df = acc_df.rename(columns={'mean' : mean_output_label, 
                                    'std' : std_output_label,
                                    'mean_high' : mean_correct_high_name, 
                                    'std_high' : std_correct_high_name, 
                                    'mean_low' : mean_correct_low_name, 
                                    'std_low' : std_correct_low_name
                                   })
    #print(acc_df)
    return acc_df
    
    
def compute_correctness_all(df, out_dir = '.', strain_col=Names.STRAIN, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL, logger=l):
    drop_list = ['Time', 'FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H',
                      'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W', 'live', 'index']
    drop_list = [x for x in drop_list if x in df.columns]
    #if 'index' in df.columns:
    #    drop_list.append('index')
    result = df.drop(drop_list,
                      axis=1).drop_duplicates().reset_index()

    if 'plan' in df.columns:
        experiment = df['plan'].unique()[0]
    elif 'experiment_id' in df.columns:
        experiment = df['experiment_id'].unique()[0]

    logger.info("Compute Correctness for experiment " + str(experiment))
        
    results = []
    # Get correctness w/o dead cells gated
    logger.debug("Computing Threshold, no gating ...")
    results.append(compute_correctness(df,
                                    output_label='probability_correct_threshold',
                                    strain_col=strain_col,
                                    high_control=high_control,
                                    low_control=low_control,
                                    mean_name='mean_log_gfp',
                                    std_name='std_log_gfp',
                                    mean_correct_name='mean_correct_threshold',
                                    std_correct_name='std_correct_threshold',
                                    mean_correct_high_name='mean_correct_high_threshold',
                                    std_correct_high_name='std_correct_high_threshold',
                                    mean_correct_low_name='mean_correct_low_threshold',
                                    std_correct_low_name='std_correct_low_threshold',
                                    count_name='count',
                                    threshold_name='threshold',
                                    logger=logger))
    logger.debug("Computing Random Forest, no gating ...")
    results.append(compute_correctness_classifier(df,
                                            out_dir = out_dir,
                                            mean_output_label='mean_correct_classifier',
                                            std_output_label='std_correct_classifier',
                                            mean_correct_high_name='mean_correct_high_classifier',
                                            std_correct_high_name='std_correct_high_classifier',
                                            mean_correct_low_name='mean_correct_low_classifier',
                                            std_correct_low_name='std_correct_low_classifier',
                                            description = str(experiment)+"_correctness",
                                            high_control=high_control,
                                            low_control=low_control,
                                            strain_col=strain_col,
                                            use_harness=True,
                                            logger=logger))

    # Get correctness w/ dead cells gated
    if 'live' in df.columns:
        gated_df = df.loc[df['live'] == 1]
        logger.debug("Computing Threhold, with gating ...")
        results.append(compute_correctness(gated_df,
                                        output_label='probability_correct_threshold',
                                        high_control=high_control,
                                        low_control=low_control,
                                        strain_col=strain_col,
                                        mean_name='mean_log_gfp_live',
                                        std_name='std_log_gfp_live',
                                        mean_correct_name='mean_correct_threshold_live',
                                        std_correct_name='std_correct_threshold_live',
                                        mean_correct_high_name='mean_correct_high_threshold_live',
                                        std_correct_high_name='std_correct_high_threshold_live',
                                        mean_correct_low_name='mean_correct_low_threshold_live',
                                        std_correct_low_name='std_correct_low_threshold_live',
                                        count_name='count_live',
                                        threshold_name='threshold_live',
                                        logger=logger))

        logger.debug("Computing Random Forest, with gating ...")
        results.append(compute_correctness_classifier(gated_df,
                                                out_dir = out_dir,
                                                mean_output_label='mean_correct_classifier_live',
                                                std_output_label='std_correct_classifier_live',
                                                description = str(experiment) + "_correctness_live",
                                                mean_correct_high_name='mean_correct_high_classifier_live',
                                                std_correct_high_name='std_correct_high_classifier_live',
                                                mean_correct_low_name='mean_correct_low_classifier_live',
                                                std_correct_low_name='std_correct_low_classifier_live',
                                                high_control=high_control,
                                                low_control=low_control,
                                                strain_col=strain_col,
                                                use_harness=True,
                                                logger=logger))

    #print(len(result))
    for r in results:
        if 'id' in r.columns:
            result = result.merge(r, on='id')
        #print(len(result))
    return result

def write_correctness(data_file, overwrite, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL, strain_col=Names.STRAIN, logger=l):
    try:
        data_dir = "/".join(data_file.split('/')[0:-1])
        out_path = os.path.join(data_dir, 'correctness')
        out_file = os.path.join(out_path, data_file.split('/')[-1])
        if overwrite or not os.path.isfile(out_file):
            logger.info("Computing correctness file: " + out_file + ", strain_col = " + strain_col)
            data_df = pd.read_csv(data_file,dtype={'od': float, 'input' : object, 'output' : object}, index_col=0 )
            correctness_df = compute_correctness_all(data_df, out_dir=out_path, 
                                                     high_control=high_control, low_control=low_control,
                                                     strain_col=strain_col)
            logger.info("Writing correctness file: " + out_file)
            correctness_df.to_csv(out_file)
    except Exception as e:
        logger.warn("File failed: " + data_file + " with: " + str(e))
        pass

def write_correctness_job(args, kwargs):
    write_correctness(*args, **kwargs)

def write_correctness_files(data, overwrite=False, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL, num_processes=4, strain_col=Names.STRAIN, logger=l):
    import multiprocessing
    #pool = multiprocessing.Pool(int(multiprocessing.cpu_count()))
    pool = multiprocessing.Pool(num_processes)
    multiprocessing.cpu_count()
    tasks = []
    for d in data:       
        tasks.append(((d, overwrite), {'high_control':high_control, 'low_control':low_control, 'strain_col' : strain_col}))
#        tasks.append((d, overwrite))
    results = [pool.apply_async(write_correctness_job, t) for t in tasks]

    for result in results:
        data_list = result.get()



def compute_correctness_with_classifier(df, 
                                     out_dir = '.', 
                                     high_control=Names.NOR_00_CONTROL, 
                                     low_control=Names.WT_LIVE_CONTROL,
                                     add_predictions = False,
                                  use_harness=True):
    drop_list = ['Time', 'FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H',
                      'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W', 'live']
    if 'index' in df.columns:
        drop_list.append('index')
    result = df.drop(drop_list,
                      axis=1).drop_duplicates().reset_index()

    experiment = df['plan'].unique()[0]


    results = []

    l.debug("Computing Random Forest, no gating ...")
    results.append(compute_correctness_classifier(df,
                                            out_dir = out_dir,
                                            mean_output_label='mean_correct_classifier',
                                            std_output_label='std_correct_classifier',
                                            description = experiment+"_correctness",
                                            high_control=high_control,
                                            low_control=low_control,
                                             add_predictions = add_predictions,
                                                 use_harness=use_harness))
    

    #print(len(result))
    for r in results:
        result = result.merge(r, on='id')
        #print(len(result))
    return result



