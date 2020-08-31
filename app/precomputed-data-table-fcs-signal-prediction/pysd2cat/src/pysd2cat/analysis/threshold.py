import pandas as pd
import math
import numpy as np
import os
from pysd2cat.data import pipeline
from pysd2cat.analysis.Names import Names    

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)
 

def get_experiment_correctness_and_metadata(adf):
    """
    Cleanup the correctness dataframe from an experiment.
    """
    def media_fix(x):
        media_map = {
            "SC Media" : "standard_media",
            "SC+Oleate+Adenine" : "standard_media",
            "Synthetic_Complete" : "standard_media",
            "standard_media" : "standard_media",

            "Synthetic_Complete_2%Glycerol_2%Ethanol" : "slow_media",
            "SC Slow" : "slow_media",
            "slow_media" : "slow_media",
            
            "Synthetic_Complete_1%Sorbitol" : "high_osm_media",
            "SC High Osm" : "high_osm_media", 
            "high_osm_media" : "high_osm_media",
            
            
            "YPAD" : "rich_media",
            "Yeast_Extract_Peptone_Adenine_Dextrose (a.k.a. YPAD Media)" : "rich_media",
            "rich_media" : "rich_media"
        }
        if type(x) is str:
            return media_map[x]
        else:
            return x
        
    def fix_temp(x):
        if type(x['inc_temp']) is str:
            x['inc_temp'] = float(x['inc_temp'].split("_")[1])
        return x

    def fix_time(x):
        if 'inc_time_2' not in x or x['inc_time_2'] is None:
            x['inc_time_2'] = 18
        elif type(x['inc_time_2']) is str:
            x['inc_time_2'] = float(x['inc_time_2'].split(":")[0])

        return x


    
    drop_list = ['Unnamed: 0', 'FSC_A',
           'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W',
           'SSC_W', 'BL1_W', 'RL1_W', 'Time']
    final_df = adf
    #print(final_df['media'])
    #final_df.loc[:, 'media'] = final_df['media'].apply(media_fix)
    final_df = final_df.apply(fix_input, axis=1)            
    final_df = final_df.apply(fix_output, axis=1)
    final_df = final_df.apply(fix_temp, axis=1)
    final_df = final_df.apply(fix_time, axis=1)


    #print(final_df['media'])
    
    return final_df

def get_sample_correctness(data):
    """
    Get a dataframe that includes the correctness of all samples and condition sets.
    Requires dataframes for correctness are present in paths listed in 'data' parameter.
    """
    all_df = pd.DataFrame()
    for d in data:
        if os.path.isfile(d):
            #print("Getting Correctness for: " + str(d))
            #df = pd.read_csv(d)
            adata = os.path.join("/".join(d.split('/')[0:-1]), 'correctness', d.split('/')[-1])
            if os.path.isfile(adata):
                adf = pd.read_csv(adata, dtype={'od': float, 'input' : object}, index_col=0)
                if 'media' in adf.columns:
                    final_df = get_experiment_correctness_and_metadata(adf)
                    all_df = all_df.append(final_df, ignore_index=True)
    #all_df.loc[:, 'prc_improve'] = all_df['probability_correct_live'] - all_df['probability_correct']
    #all_df.loc[:, 'live_proportion'] = all_df['count_live'] / all_df['count']
    #all_df['sample_time'] = all_df.apply(pipeline.get_sample_time, axis=1)
    return all_df




def get_threshold(df, channel='BL1_A', strain_col=Names.STRAIN, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL, logger=l):

    if False and high_control not in df[strain_col].unique():
        fixed_high_control = high_control.replace(" ", "-")
        if fixed_high_control in df[strain_col].unique():
            high_control = fixed_high_control
        else:
            raise Exception("Cannot compute threshold if do not have both low and high control for high_control=\"" + str(high_control) + "\" low_control = \"" + str(low_control) + "\" Have strain_name's: " + str(df[strain_col].unique()))
    if False and low_control not in df[strain_col].unique():
        fixed_low_control = low_control.replace(" ", "-")
        if fixed_low_control in df[strain_col].unique():
            low_control = fixed_low_control
        else:
            raise Exception("Cannot compute threshold if do not have both low and high control for high_control=\"" + str(high_control) + "\" low_control = \"" + str(low_control) + "\" Have strain_name's: " + str(df[strain_col].unique()))


           
    ## Prepare the data for high and low controls
    high_df = df.loc[( df[strain_col] == high_control)]
    high_df.loc[:,'output'] = high_df.apply(lambda x: 1, axis=1)
    low_df = df.loc[(df[strain_col] == low_control) ]
    low_df.loc[:,'output'] = low_df.apply(lambda x: 0, axis=1)
    high_low_df = high_df.append(low_df)
    high_low_df = high_low_df.loc[high_low_df[channel] > 0]
    high_low_df.loc[:, channel] = np.log(high_low_df[channel]).replace([np.inf, -np.inf], np.nan).dropna()
    #high_low_df[channel]
    
    if len(high_df) == 0 or len(low_df) == 0:
        raise Exception("Cannot compute threshold if do not have both low and high control for high_control=\"" + str(high_control) + "\" low_control = \"" + str(low_control) + "\" Have strain_name's: " + str(df[strain_col].unique()))

    ## Setup Gradient Descent Paramters

    cur_x = high_low_df[channel].mean() # The algorithm starts at mean
    #print("Starting theshold = " + str(cur_x))
    rate = 0.00001 # Learning rate
    precision = 0.0001 #This tells us when to stop the algorithm
    previous_step_size = 1 #
    max_iters = 100 # maximum number of iterations
    iters = 0 #iteration counter

    def correct_side(threshold, value, output):
        if output == 1 and value > threshold:
            return 1
        elif output == 0 and value <= threshold:
            return 1
        else:
            return 0

    def gradient(x):
        delta = 0.1
        xp = x + delta
        correct = high_low_df.apply(lambda row : correct_side(x, row['BL1_A'], row['output']), axis=1)
        correctp = high_low_df.apply(lambda row : correct_side(xp, row['BL1_A'], row['output']), axis=1)
        # print(sum(correct))
        # print(sum(correctp))
        try:
            grad = (np.sum(correct) - np.sum(correctp))/delta
        except Exception as e:
            logger.warn(sum(correct))
            logger.warn(sum(correctp))
            logger.warn(e)
        #print("Gradient at: " + str(x) + " is " + str(grad))
        return grad

    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x - rate * gradient(prev_x) #Grad descent
        previous_step_size = abs(cur_x - prev_x) #Change in x
        iters = iters+1 #iteration count
        #print("Iteration",iters,"\nX value is",cur_x) #Print iterations

    max_value = sum(high_low_df.apply(lambda row : correct_side(cur_x, row['BL1_A'], row['output']), axis=1))/len(high_low_df)
    # print("Maximized at: " + str(cur_x))
    # print("Value at Max: " + str(max_value))
    return cur_x, max_value


def fix_input(row):
    if row['input'] == '1.0':
        row['input'] = '01'
    elif row['input'] == '0.0':
        row['input'] = '00'
    elif row['input'] == '10.0':
        row['input'] = '10'
    elif row['input'] == '11.0':
        row['input'] = '11'
    return row

def fix_output(row):
    if row['output'] == '1.0':
        row['output'] = '1'
    elif row['output'] == '0.0':
        row['output'] = '0'
    return row


def compute_correctness(m_df,
                        id_name="id",
                     channel='BL1_A',
                     thresholds=None,
                     use_log_value=True,
                     high_control=Names.NOR_00_CONTROL,
                     low_control=Names.WT_LIVE_CONTROL,
                     strain_col=Names.STRAIN,
                     output_label='probability_correct',
                     mean_name='mean_log_gfp',
                     std_name='std_log_gfp',
                     mean_correct_name='probability_correct',
                     std_correct_name='std_correct',
                     mean_correct_high_name='mean_correct_high_threshold',
                     std_correct_high_name='std_correct_high_threshold',
                     mean_correct_low_name='mean_correct_low_threshold',
                     std_correct_low_name='std_correct_low_threshold',
                     count_name='count',
                     threshold_name='threshold',
                     logger=l
                     ):
    if thresholds is None:
        try:
            threshold, threshold_quality = get_threshold(m_df, channel, strain_col=strain_col, high_control=high_control, low_control=low_control)
            thresholds = [threshold]
        except Exception as e:
            #print(e)
            raise Exception("Could not find controls to auto-set threshold: " + str(e))
            #thresholds = [np.log(10000)]

    #print("Threshold  = " + str(thresholds[0]))
    samples = m_df[id_name].unique()
    logger.info("samples length: {}".format(len(samples)))
    plot_df = pd.DataFrame()
    for sample_id in samples:
                
        #print("sample_id: {}".format(sample_id))
        sample = m_df.loc[m_df[id_name] == sample_id]
        if len(sample['output'].dropna().unique()) == 0:
            continue
            
        #print("sample.head(): {}".format(sample.head()))
        circuit = sample['gate'].unique()[0]
        #print("circuit: {} type: {}".format(circuit, type(circuit)))
        if circuit:
            value_df = sample[[channel, 'output']].rename(index=str, columns={channel: "value"})
            if use_log_value:
                value_df = value_df.loc[value_df['value'] > 0]
                value_df.loc[:,'value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()
            #print(value_df.head())
            thold_df = do_threshold_analysis(value_df,
                                             thresholds,
                                             mean_correct_name=mean_correct_name,
                                             std_correct_name=std_correct_name,
                                             mean_correct_high_name=mean_correct_high_name,
                                             std_correct_high_name=std_correct_high_name,
                                             mean_correct_low_name=mean_correct_low_name,
                                             std_correct_low_name=std_correct_low_name,
                                             count_name=count_name,
                                             threshold_name=threshold_name)

            
            thold_df[mean_name] = np.mean(value_df['value'])
            thold_df[std_name] = np.std(value_df['value'])
        else:
            thold_df = {}
            
        thold_df[id_name] = sample_id
        
#            for i in ['gate', 'input', 'output', 'od', 'media',
#                      'inc_temp', 'replicate', 'inc_time_1',
#                      'inc_time_2', 'strain_name']:
#                if i in sample.columns:
#                    thold_df[i] = sample[i].unique()[0]
#                elif i == 'inc_temp':
#                    thold_df[i] = 'warm_30'
#                else:
#                    thold_df[i] = None
#            thold_df = thold_df.apply(fix_input, axis=1)            
#            thold_df = thold_df.apply(fix_output, axis=1)
            
            ## if 'live' in m_df.columns:
            ##     sample_live = sample.loc[sample['live'] == 1]
            ##     value_df = sample_live[[channel, 'output']].rename(index=str, columns={channel: "value"})
            ##     if use_log_value:
            ##         value_df = value_df.loc[value_df['value'] > 0]
            ##         value_df['value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()

            ##     #print(value_df.shape())
            ##     thold_live_df = do_threshold_analysis(value_df, thresholds)
            ##     thold_df['probability_correct_live'] = thold_live_df['probability_correct']
            ##     thold_df['standard_error_correct_live'] = thold_live_df['standard_error_correct']
            ##     thold_df['count_live'] = thold_live_df['count']
            ##     thold_df['mean_log_gfp_live'] = np.mean(value_df['value'])
            ##     thold_df['std_log_gfp_live'] = np.std(value_df['value'])
            ## else:
            ##     thold_df['probability_correct_live'] = None #thold_df['probability_correct']
            ##     thold_df['standard_error_correct_live'] = None #thold_df['standard_error_correct']
            ##     thold_df['count_live'] = None #thold_df['count']
            ##     thold_df['mean_log_gfp_live'] = None
            ##     thold_df['std_log_gfp_live'] = None



                
            #print(thold_df)
        plot_df = plot_df.append(thold_df, ignore_index=True)
    #plot_df = plot_df.rename(columns={mean_correct_name : output_label})
    return plot_df 


def compute_correctness2(m_df,
                        id_name="id",
                     channel='BL1_A',
                     threshold=None,
                     use_log_value=True,
                     high_control=Names.NOR_00_CONTROL,
                     low_control=Names.WT_LIVE_CONTROL,
                     output_label='probability_correct',
                     mean_name='mean_log_gfp',
                     std_name='std_log_gfp',
                     mean_correct_name='probability_correct',
                     std_correct_name='std_correct',
                     mean_correct_high_name='mean_correct_high_threshold',
                     std_correct_high_name='std_correct_high_threshold',
                     mean_correct_low_name='mean_correct_low_threshold',
                     std_correct_low_name='std_correct_low_threshold',
                     count_name='count',
                     threshold_name='threshold'
                     ):
    if threshold is None:
        try:
            threshold, threshold_quality = get_threshold(m_df, channel, high_control=high_control, low_control=low_control)
        except Exception as e:
            #print(e)
            raise Exception("Could not find controls to auto-set threshold: " + str(e))
            #thresholds = [np.log(10000)]

    #print("Threshold  = " + str(thresholds[0]))
    #samples = m_df[id_name].unique()
    #print("samples length: {}".format(len(samples)))
    #plot_df = pd.DataFrame()
    
    prediction_df = pd.DataFrame()
    
    def correct(x, threshold):
        if x['output'].isnull():
            return None
        
        true_gate_output = int(x['output'])
        measured_gate_output = float(row['value'])
        if (true_gate_output == 1 and measured_gate_output >= threshold) or \
           (true_gate_output == 0 and measured_gate_output < threshold) :
            return 1.0
        else:
            return 0.0
        
    def high(x, threshold):
        if x['output'].isnull():
            return None
        
        true_gate_output = int(x['output'])
        measured_gate_output = float(row['value'])
        if measured_gate_output >= threshold:
            return 1.0
        else:
            return 0.0

    def low(x, threshold):
        if x['output'].isnull():
            return None
        
        true_gate_output = int(x['output'])
        measured_gate_output = float(row['value'])
        if measured_gate_output < threshold:
            return 1.0
        else:
            return 0.0

    value_df = m_df[[channel, 'output', id_name]].rename(index=str, columns={channel: "value"})
    if use_log_value:
        value_df = value_df.loc[value_df['value'] > 0]
        value_df.loc[:,'value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()


        
    prediction_df.loc[:, id_name] = value_df[id_name]
    prediction_df.loc[:, 'correct'] = value_df.apply(lambda x: correct(x, threshold), axis=1)
    prediction_df.loc[:, 'high'] = value_df.apply(lambda x: high(x, threshold), axis=1)
    prediction_df.loc[:, 'low'] = value_df.apply(lambda x: low(x, threshold), axis=1)
    
    thold_df.loc[:, mean_correct_name] = predictions_df.groupby([id_name])['correct'].agg('mean').reset_index()
    
    return thold_df


def do_threshold_analysis(df,
                          thresholds,
                          mean_correct_name='probability_correct',
                          std_correct_name='std_correct',
                          mean_correct_high_name='mean_correct_high_threshold',
                          std_correct_high_name='std_correct_high_threshold',
                          mean_correct_low_name='mean_correct_low_threshold',
                          std_correct_low_name='std_correct_low_threshold',
                          count_name='count',
                          threshold_name='threshold'
                          ):
    """
    Get Probability that samples fall on correct side of threshold
    """
    count = 0
    correct = []
    low = []
    high = []
    for idx, threshold in enumerate(thresholds):
        correct.append(0)
        low.append(0)
        high.append(0)

    #print("df columns: {}".format(df.columns))
    for idx, row in df.iterrows():
        true_gate_output = int(row['output'])
        measured_gate_output = float(row['value'])
        count = count + 1
        #print("count: {} true_gate_output: {} measured_gate_output: {} threshold: {}".format(count, true_gate_output, measured_gate_output, threshold))
        for idx, threshold in enumerate(thresholds):
            #print(str(true_gate_output) + " " + str(measured_gate_output))
            if (true_gate_output == 1 and measured_gate_output >= threshold) or \
               (true_gate_output == 0 and measured_gate_output < threshold) :
                correct[idx] = correct[idx] + 1
            if measured_gate_output >= threshold:
                high[idx] = high[idx] + 1
            if measured_gate_output < threshold:
                low[idx] = low[idx] + 1


    #print("correct length: {} correct[0]: {} count: {}".format(len(correct), correct[0], count))
    results = pd.DataFrame()
    for idx, threshold in enumerate(thresholds):
        if count > 0:
            pr = correct[idx] / float(count)
            se = math.sqrt(pr*(1-pr)/float(count))
            low_pr = low[idx] /float(count)
            low_se = math.sqrt(low_pr*(1-low_pr)/float(count))
            high_pr = high[idx] /float(count)
            high_se = math.sqrt(high_pr*(1-low_pr)/float(count))
        else:
            pr = 0
            se = 0
            low_pr = 0
            low_se = 0
            high_pr = 0
            high_se = 0

        #print("count: {} idx: {} threshold: {} pr: {}".format(count, idx, threshold, pr))
        results= results.append({
            mean_correct_name : pr, 
            std_correct_name : se,
            mean_correct_high_name : high_pr,
            std_correct_high_name : high_se,
            mean_correct_low_name : low_pr,
            std_correct_low_name : low_se,
            count_name : count,
            threshold_name : threshold}, ignore_index=True)
    return results
    

    

