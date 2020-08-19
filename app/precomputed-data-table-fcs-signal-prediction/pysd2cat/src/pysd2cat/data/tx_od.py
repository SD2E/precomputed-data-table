from pysd2cat.data import pipeline
import pandas as pd
import os
import glob
from os.path import expanduser
import transcriptic
import csv
import logging
import json
import math
    
import transcriptic 
from transcriptic import commands, Container
from transcriptic.jupyter import objects

from pysd2cat.xplan import experiment_request

    
    
############################################################################
## ETL related data functions available from pipeline
############################################################################

def get_project_run_groups(project_id):
    """
    Given a project_id, return a dataframe that has a row for each experiment.
    Each row provides the run id of part 1, part 2, and calibration. 
    """
    
    project = transcriptic.project(project_id)
    try:
        project_runs = project.runs()
    except Exception as e:
        return pd.DataFrame()
        
    
    #print(project_runs)
    
    ## Get calibration run
    calibration_runs = project_runs[project_runs['Name'].str.contains('Plate Reader Calibration')]
    if len(calibration_runs) > 0:
        calibration_run = calibration_runs['id'].unique()[0]
    else:
        calibration_run = None

    part_one_runs = project_runs[project_runs['Name'].str.contains('YeastGatesPartI_')]
    
    
    def strip_hr(x):
        if x['Name'].endswith("hr") or x['Name'].endswith("h"):
            x['Name'] = "_".join(x['Name'].split("_")[0:-1])
        return x
    
    part_one_runs.loc[:,'id'] = part_one_runs.apply(strip_hr, axis=1)
    
    part_one_runs.loc[:,'index'] = part_one_runs.apply(lambda x: x['Name'].split('_')[-1], axis=1)
    part_one_runs = part_one_runs.rename(columns={'id' : 'part_1_id'}).drop(columns=['Name'])
    #print(part_one_runs)

    part_two_runs = project_runs[project_runs['Name'].str.contains('YeastGatesPartII')]
    part_two_runs.loc[:,'id'] = part_two_runs.apply(strip_hr, axis=1)
    
    if len(part_two_runs) > 0:
        part_two_runs.loc[:,'index'] = part_two_runs.apply(lambda x: x['Name'].split('_')[-1], axis=1)
        part_two_runs = part_two_runs.rename(columns={'id' : 'part_2_id'}).drop(columns=['Name'])
        #print(part_two_runs)
        both_parts = part_one_runs.merge(part_two_runs, on="index").drop(columns=['index'])
    else:
        both_parts = part_one_runs.drop(columns=['index'])
        both_parts.loc[:, 'part_2_id'] = None
        
    both_parts.loc[:, 'calibration_id'] = calibration_run
    both_parts.loc[:, 'project_id'] = project_id
    return both_parts

def get_run_groups(projects, project_runs_file):
    """
    Cache all run groups in projects into project_runs_file.  Before cache exists,
    this call can be costly due to the necessary data retrieval.
    """
    if os.path.exists(project_runs_file):
        df = pd.read_csv(project_runs_file, index_col=0)
    else:
        df = pd.DataFrame()
                
    for project_id in projects:
        #print(project_id)
        project_id = project_id.split('/')[-1]
        if project_id not in df['project_id'].unique():
            df = df.append(get_project_run_groups(project_id), ignore_index = True)
    df = df.replace({pd.np.nan: None})
    df.to_csv(project_runs_file)
    return df



def well_idx_to_well_id(idx):
    """
    Convert an well index in a 96-well plate (8x12) to [char][int] ID.
    0 -> a1
    ...
    95 -> h12
    """
    assert idx >= 0 and idx < (12 * 8), "Index not in 96-well plate"
    # a-h
    chars = [chr(ord('a') + x) for x in range(ord('h') - ord('a') + 1)]
    row, col = divmod(idx, 12)
    return '{}{}'.format(chars[row], col + 1)





def get_stocks(run_obj):
    
    def get_stock_from_str(st):
        # Strip prefix
        sp = "-".join(st.split('-')[1:])

        #Convert '_' to '-'
        sp = sp.replace('_', '-')

        stock_plate = "-".join(sp.split('-')[:-1])
        glycerol_plate_index = sp.split('-')[-1]

        return stock_plate, glycerol_plate_index


    container_names = run_obj.containers.Name
    #print(container_names)
    stock_plates = [x for x in container_names if "YeastGatesStockPlate" in x]
    #print(stock_plates)
    if len(stock_plates) == 1:
        glycerol_stock, glycerol_plate_index = get_stock_from_str(stock_plates[0])
    else:
        glycerol_stock = None
        glycerol_plate_index = None
        
    #print(glycerol_stock)
    #print(plate_index)
    return glycerol_stock, glycerol_plate_index


def create_od_csv(run_id, csv_path='tmp.csv'):
    """
    Write a CSV file at path for OD data associated with run_id. 
    The OD data is uncorrected and adds some minimal amount of metadata
    needed to merge it with more verbose metadata.
    """
    run_obj = get_tx_run(run_id)
    d = run_obj.data
        

    glycerol_stock, glycerol_plate_index = get_stocks(run_obj)    
        
    is_calibration = False    
    measurements = {}

    # Ensure the absorbance dataset always comes before fluorescence
    # in the iteration. Logic below depends on this.
    datasets = sorted(d['Datasets'], key=lambda x: x.operation)
    for dataset in d['Datasets']:
        is_calibration = 'container' in dataset.attributes and 'CalibrationPlate' in dataset.attributes['container']['label']
        if dataset.data_type == "platereader" and dataset.operation == "absorbance":

            # Collect OD data
            for well, values in dataset.raw_data.items():
                if well in measurements:
                    measurements[well]['od'] =  values[0]
                else:
                    measurements[well] = {'od': values[0]}

            # Collect sample uris for each well
            for alq in dataset.attributes['container']['aliquots']:
                well = well_idx_to_well_id(alq['well_idx'])
                if well not in measurements:
                    continue
                try:
                    if is_calibration:
                        measurements[well]['sample'] = alq['name'] + "_" + str(alq['volume_ul'])
                    else:
                        if glycerol_stock and glycerol_plate_index:
                            measurements[well]['glycerol_stock'] = glycerol_stock
                            measurements[well]['glycerol_plate_index'] = glycerol_plate_index
                        #print(alq)
                        if 'control' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['control']
                        elif 'Sample_ID' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['Sample_ID']
                        if 'SynBioHub URI' in alq['properties']:
                            measurements[well]['SynBioHub URI'] = alq['properties']['SynBioHub URI']
                        else:
                            measurements[well]['SynBioHub URI'] = None
                except KeyError:
                    raise KeyError('Found sample for well which did not have OD measurement')

        elif dataset.data_type == "platereader" and dataset.operation == "fluorescence":
            for well, values in dataset.raw_data.items():
                if well in measurements:
                    measurements[well]['gfp'] =  values[0]
                else:
                    measurements[well] = {'gfp': values[0]}
                    
            for alq in dataset.attributes['container']['aliquots']:
                well = well_idx_to_well_id(alq['well_idx'])
                if well not in measurements:
                    continue
                try:
                    if is_calibration:
                        measurements[well]['sample'] = alq['name'] + "_" + str(alq['volume_ul'])
                    else:
                        if glycerol_stock and glycerol_plate_index:
                            measurements[well]['glycerol_stock'] = glycerol_stock
                            measurements[well]['glycerol_plate_index'] = glycerol_plate_index
                        
                        if 'control' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['control']
                        elif 'Sample_ID' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['Sample_ID']
                        if 'SynBioHub URI' in alq['properties']:
                            measurements[well]['SynBioHub URI'] = alq['properties']['SynBioHub URI']        
                        else:
                            measurements[well]['SynBioHub URI'] = None
                except KeyError:
                    raise KeyError('Found sample for well which did not have Fluorescence measurement')

    print(measurements)
    with open(csv_path, 'w+') as csvfile:
        if glycerol_stock and glycerol_plate_index:
            writer = csv.DictWriter(csvfile, fieldnames=['well', 'od', 'gfp', 'sample', 'SynBioHub URI', 'glycerol_stock', 'glycerol_plate_index'])
        else:
            writer = csv.DictWriter(csvfile, fieldnames=['well', 'od', 'gfp', 'sample', 'SynBioHub URI'])


        writer.writeheader()
        writer.writerows((_measurement_to_csv_dict(well, data)
                          for well, data in sorted(measurements.items())))


def get_tx_run(run_id):
    """
    Create a TX run object from run id.
    """
    
    try:
        # In the reactor, the connection will already have been
        # made at this point. Don't worry if this fails. If the
        # connection failed up above, the Run instantion will
        # still cause an exception.
        connect()
    except Exception:
        pass
    return objects.Run(run_id)

def get_tx_project(project_id):
    """
    Create a TX project object from project id.
    """
    
    try:
        # In the reactor, the connection will already have been
        # made at this point. Don't worry if this fails. If the
        # connection failed up above, the Run instantion will
        # still cause an exception.
        connect()
    except Exception:
        pass
    return objects.Project(project_id)



def _measurement_to_csv_dict(well, data):
    """
    Helper function to convert measurement created in create_od_csv to a csv row
    """
    res = data.copy()
    res['well'] = well
    return res

def get_od_scaling_factor(calibration_df):
    """
    Compute scaling factor from controls.
    """
    
    ludox_df = calibration_df.loc[calibration_df['sample'] == 'ludox_cal_100.0']
    water_df = calibration_df.loc[calibration_df['sample'] == 'water_cal_100.0']
    
    ludox_mean = ludox_df['od'].mean()
    water_mean = water_df['od'].mean()
    
    corrected_abs600 = ludox_mean - water_mean
    reference_od600 = 0.063
    scaling_factor = reference_od600 / corrected_abs600

    return scaling_factor


def make_experiment_corrected_df(part_1_df, part_2_df, calibration_df=None):
    """
    Join the run data to get pre and post ODs, then add corrections of 
    said ODs.
    """
    
    part_1_df = part_1_df.rename(columns={"od" : "pre_od_raw", "gfp" : "pre_gfp_raw", "well" : "pre_well"})
    part_2_df = part_2_df.rename(columns={"od" : "post_od_raw", "gfp" : "post_gfp_raw", "well" : "post_well"})
    #print(part_1_df)
    #print(part_2_df)
    experiment_df = part_1_df.merge(part_2_df, how='inner', on=['sample', 'SynBioHub URI']).drop(columns=['sample']).dropna().reset_index(drop=True)

    if calibration_df is not None:
        od_scaling_factor = get_od_scaling_factor(calibration_df)

        #print(experiment_df)

        experiment_df.loc[:, 'pre_od_corrected'] = experiment_df.apply(lambda x: x['pre_od_raw'] * od_scaling_factor, axis=1 )
        experiment_df.loc[:, 'post_od_corrected'] = experiment_df.apply(lambda x: x['post_od_raw'] * od_scaling_factor, axis=1 )
    else:
        #print(experiment_df)
        experiment_df.loc[:, 'pre_od_corrected'] = None
        experiment_df.loc[:, 'post_od_corrected'] = None
        
    return experiment_df

def is_yg_run(experiment):
    part_1_run = get_tx_run(experiment['part_1_id'])
    cs = part_1_run.containers
    ygsp = cs.loc[cs['Name'].str.contains('YeastGatesStockPlate')]
    return len(ygsp) > 0
 

def get_experiment_data(experiment, out_dir, overwrite=False):
    """
    For a triple of runs (part 1, part 2, and calibration), generate a
    dataframe with OD data for all wells.
    """
    
    if not is_yg_run(experiment):
        raise Exception("Cannot process this experiment type")


    
    if not pd.isna(experiment['calibration_id']):       
        calibration_file = os.path.join(out_dir, experiment['calibration_id'] + '.csv')
        if overwrite or not os.path.exists(calibration_file):
            create_od_csv(experiment['calibration_id'], csv_path=calibration_file)
        calibration_df = pd.read_csv(calibration_file, index_col = False)
    else:
        calibration_df = None

    
    part_1_file = os.path.join(out_dir, experiment['part_1_id'] + '.csv')
    part_2_file = os.path.join(out_dir, experiment['part_2_id'] + '.csv')
        
    if overwrite or not os.path.exists(part_1_file):
        create_od_csv(experiment['part_1_id'], csv_path=part_1_file)
    if overwrite or not os.path.exists(part_2_file):
        create_od_csv(experiment['part_2_id'], csv_path=part_2_file)

    part_1_df = pd.read_csv(part_1_file, index_col = False)
    part_2_df = pd.read_csv(part_2_file, index_col = False)
        
    #print(part_1_df.head())
    #print(part_2_df.head())

        
    experiment_df = make_experiment_corrected_df(part_1_df, part_2_df, calibration_df=calibration_df)
    return experiment_df
   

def get_meta(experiment, xplan_base='sd2e-projects/sd2e-project-14/'):
    """
    Get metadata for an experiment from the xplan-reactor state.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('main')

    xplan_state_file = os.path.join(expanduser("~"), xplan_base, 'xplan-reactor/state.json')
    state = json.load(open(xplan_state_file, 'r'))

    part_1_id = experiment['part_1_id'] #'r1c5vaeb8vbt9'
    #part_2_id = experiment['part_2_id'] #'r1c66mfpj7guh'

    xplan_experiment = state['runs'][part_1_id]
    #print(experiment)
    request_file = os.path.join(expanduser("~"), xplan_base, 'xplan-reactor/experiments/', xplan_experiment['experiment_id'], 'request_' + xplan_experiment['experiment_id'] + ".json")
    request = experiment_request.ExperimentRequest(**json.load(open(request_file, 'r')))
    #print("Got request")
    meta = request.pd_metadata(logger, plates=[xplan_experiment['plate_id']])
    meta = meta.rename(columns={'gate' : 'strain_circuit'})

    return meta

def get_data_and_metadata_df(experiment, out_dir, state_file, overwrite=False):
    """
    Get all metadata and data for an experiment (run triple),
    cache a copy in out_dir, and return it.
    """
    
    # Only support two part experiment right now
    if experiment['part_2_id'] is None:
        return pd.DataFrame()
    
    
    experiment_od_file = os.path.join(out_dir, experiment['part_1_id'] + "_" + experiment['part_2_id'] + '.csv')
    if overwrite or not os.path.exists(experiment_od_file):
        try:
            #print(experiment_od_file)
            data = get_experiment_data(experiment, out_dir, overwrite=overwrite) 
            #print(data.head())
            
            meta = get_meta(experiment)
            #print(meta.head())
            meta_and_data_df = meta.merge(data,  left_on='well', right_on='post_well')
            meta_and_data_df.to_csv(experiment_od_file)
        except Exception as e:
            print("Could not generate dataframe for experiment: " + str(experiment) + " because " + str(e))
            return pd.DataFrame()
    else: 
        meta_and_data_df = pd.read_csv(experiment_od_file, dtype={'input': object})

    return meta_and_data_df


