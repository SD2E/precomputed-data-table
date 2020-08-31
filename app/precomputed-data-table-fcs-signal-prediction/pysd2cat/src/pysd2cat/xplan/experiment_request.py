import json
#from xplan_api.autoprotocol import xplan_to_params
import uuid
import pandas as pd
from .experiment_request_utils import get_role
from .experiment_request_utils import get_source_for_strain, generate_well_parameters, get_container_aliquots, gate_info
from pysd2cat.data.pipeline import get_xplan_data_and_metadata_df, handle_missing_data
from pysd2cat.analysis.Names import Names

class ExperimentRequest(dict):
    """
    An experiment request for automating conversion to Transcriptic params.json
    """

    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self.uuid = str(uuid.uuid1())
#        self.samples = []
#        self.id = experiment_id

    def __str__(self):
        return json.dumps(self, indent=4, sort_keys=True, separators=(',', ': '))

    def __get_refs_for_samples(self):
        """
        Deprecated: no longer need refs for request
        """
        keys = set([])
        refs = {}
        for sample in self['samples']:
            
            #print(sample['source_container'])
            key = sample['source_container']['label']
            if not key in keys:
               # print("Adding ref for container: " + str(key))
                keys.add(key)
                container_type = sample['source_container']['container_type_id']
                store = sample['source_container']['storage_condition']
                aliquots = sample['source_container_aliquots']
                value = { "type" : container_type,
                          "name" : key,
                          "store" : store,
                          "aliquots" : aliquots
                        }
                #print(value)
                refs.update({key:value})
        return refs
                
        
    def __get_inputs_for_plate(self, plate):
        #samples = []
        #ods = [ "0.0003", "0.00075" ]
        #print("getting inputs for plate")
        src_wells = {}
        for well,well_properties in self['plates'][plate]['wells'].items():
            src_well = well_properties['src_well']

            if src_well in src_wells:
                dest_ods = src_wells[src_well]['dest_od']
            else:                
                dest_ods = []
                src_wells[src_well] = { 'source_well' : src_well, 'dest_od' : dest_ods }
            
            
            
            #for idx,od in enumerate(ods):
            well_od = {
                    "targ_od" : well_properties['od'],
                    "dest_well" : well #map_dest(well, idx)
                    }
            dest_ods.append(well_od)
            
 
        samples = [ v for k, v in src_wells.items() ]
            
        inputs = { "specify_locations" : { "samples" : samples } }
        return inputs

    def __get_experiment_set_parameters(self):
        return {
            "inc_time_1" : self['inc_time_1'], 
            "inc_time_2" : self['inc_time_2'],
            "inc_temp" : self['inc_temp'], 
            }

    def __get_experiment_plate_parameters(self, plate_id):
        return {
            "inducer" : self['plates'][plate_id]["inducer"],
            "inducer_conc" : self['plates'][plate_id]["inducer_conc"],
            "inducer_unit" : self['plates'][plate_id]["inducer_unit"],
            "fluor_ex" : self['plates'][plate_id]["fluor_ex"],
            "fluor_em" : self['plates'][plate_id]["fluor_em"],                                    
            "growth_media_1" : self['plates'][plate_id]["growth_media_1"],
            "growth_media_2" : self['plates'][plate_id]["growth_media_2"],
            "gfp_control" : self['plates'][plate_id]['gfp_control'],
            "wt_control" : self['plates'][plate_id]['wt_control'],
            "source_plate" : self['plates'][plate_id]['source_container']['id'],
            "store_growth_plate2" : self['plates'][plate_id]['store_growth_plate2'],
#            "od_cutoff" : self['plates'][plate_id]['od_cutoff']
            }
    
    def to_params(self):
        """
        Convert to Transcriptic params.json format
        """
        params_set = {}
        exp_set_params = self.__get_experiment_set_parameters()
        #print(exp_set_params)
        for plate_id, plate in self['plates'].items():
            #print(self['plates'][plate_id])
            tx_params = {}
            #tx_params['refs'] = self.__get_refs_for_samples() #xplan_to_params.generate_refs(self['resources'])
            inputs = self.__get_inputs_for_plate(plate_id)
            exp_plate_params = self.__get_experiment_plate_parameters(plate_id)
            #print(exp_plate_params)
            exp_params = {}
            exp_params.update(exp_set_params)
            exp_params.update(exp_plate_params)
            tx_params['parameters'] = {
                'exp_params' : exp_params,
                'sample_selection' : {
                    "inputs" : inputs,
                    "value" : "specify_locations"
                    }            
            }
            params_set[str(plate_id)] = json.dumps(tx_params, indent=4, sort_keys=True, separators=(',', ': '))
        #print(params_set)
        return params_set



    
    def pd_metadata(self, logger, plates=None):
        def df_empty(columns, dtypes, index=None):
            assert len(columns)==len(dtypes)
            df = pd.DataFrame(index=index)
            for c,d in zip(columns, dtypes):
                df[c] = pd.Series(dtype=d)
            return df
        
        exp_param_columns = ["inc_temp", "inc_time_1", "inc_time_2", "experiment_id"]
        exp_param_dtypes = [str, str, str, str]
        exp_param1_columns = ['lab']
        exp_param1_dtypes = [str]
        tx_plate_params_columns = ["growth_media_1", "growth_media_2", "od_cutoff", 'source_container', 'lab_id']
        tx_plate_params_dtypes = [str, str, float, str, str]
        well_columns = ['id', 'strain', 'gate', 'input', 'od', 'filename', 'replicate', 'output', 'media']
        well_dtypes = [str, str, str, str, float, str, int, str, str]
        
        columns = exp_param_columns + exp_param1_columns + tx_plate_params_columns + well_columns
        dtypes = exp_param_dtypes + exp_param1_dtypes + tx_plate_params_dtypes + well_dtypes
        
        sample_records = df_empty(columns, dtypes)
        exp_params = {}
        for param in exp_param_columns:
            exp_params.update({param : self[param]})
        exp_params['lab'] = "transcriptic"
        
        logger.debug(exp_params)
            
        for plate_id, plate in self['plates'].items():
            if plates is not None and plate_id not in plates:
                continue
            logger.debug(plate_id)
            plate_params = {}
           
            for param in tx_plate_params_columns:
                plate_params.update({param : plate[param]})
            
            
            if type(plate_params['source_container']) is dict:
                plate_params['source_container'] = plate_params['source_container']['id']
             
                
               
            plate_params['plate_id'] = plate_id
            logger.debug(plate_params)
            for well_id, well in plate['wells'].items():
                #logger.debug(well_id)
                #logger.debug(well)
                 
                
                well_params = {}
                well_params['id'] = exp_params['experiment_id'] + "_" + plate_id + "_" + well_id
                if 'strain' not in well or well['strain'] is None:
                    continue
                elif type(well['strain']) is str:
                    well_params['strain'] = well['strain']
                elif 'gate' in well['strain'] and 'value' in well['strain']['gate']:
                    well_params['strain'] = well['strain']['gate']['value']
                elif 'gate' in well['strain']:
                    well_params['strain'] = well['strain']['gate']

                if 'source' in well:
                    well_index = well['source']['index']
                    well_index_s = str(well['source']['index'])

                    ## Sometimes indices and keys don't have same type :(
                    if well_index in well['source']['Gate']:
                        well_params['gate'] = well['source']['Gate'][well_index]
                    elif well_index_s in well['source']['Gate']:
                        well_params['gate'] = well['source']['Gate'][well_index_s]

                    strain_id = well_params['strain'].split("/")[-2:-1]
                    well_params['input'] = gate_info({ 'gate' : strain_id})['input']
                    if isinstance(well_params['input'], list):
                        # Make inputs hashable.
                        well_params['input'] = "".join([str(s) for s in well_params['input']])
                    well_params['input'] = str(well_params['input'])
                if 'od' in well:
                    well_params['od'] = well['od']
                if 'measurements' in well and well['measurements'] is not None:    
                    well_params['filename'] = well['measurements'][0]
                if 'replicate' in well:
                    well_params['replicate'] = int(well['replicate'])
                well_params['output'] = handle_missing_data(well_params, Names.OUTPUT)
                well_params['well'] = well_id.lower()    

                plate_params['media'] = plate_params['growth_media_2']
                sample_record = {}
                for param_list in [ exp_params, plate_params, well_params ]:
                    #print(param_list)
                    for k, v in param_list.items():
                        sample_record[k] = v                        
                sample_records = sample_records.append(sample_record, ignore_index=True)

        #def zero_index_replicates(row):
        #    row['replicate'] = row['replicate'] - 1
        #    return row
        #replicates = sample_records.replicate.unique()
        #robj.logger.info("Replicate ids: " + str(replicates))
        #if 0 not in replicates:
        #    sample_records = sample_records.apply(zero_index_replicates, axis=1)
        
        return sample_records
            

    def to_pd(self, logger, data_dir,  plates=None, fraction=None, max_records=None):
        #print("to_pd()")
        metadata_df = self.pd_metadata(logger, plates)
        logger.info(metadata_df.strain.unique())
        logger.info("Getting data...")
        try:
            df = get_xplan_data_and_metadata_df(metadata_df, data_dir, fraction=fraction, max_records=max_records)
            #df = metadata_df
        except Exception as e:
            logger.info("Could not get data because: " + str(e))
            
        logger.info(df.head())
        return df


 

    def generate_well_parameters_from_tx(self, source_container, aliquot, tx_properties):
        print(tx_properties)
        if 'SynBioHub URI' in tx_properties and 'Gate' in tx_properties:
            strain = { 'gate' : tx_properties['SynBioHub URI'], 'role' : get_role(tx_properties['Gate'])}
        else:
            strain = None
        if 'TargetOD' in tx_properties:
            od = tx_properties['TargetOD']
        else:
            od = None

        if strain is not None:
            source = get_source_for_strain(source_container.aliquots, strain)
        else:
            source = None
        if source is not None:
            src_well = source_container.container_type.humanize(source['index'])
        else:
            src_well = None
        
        if 'replicate' in tx_properties:
            replicate = int(tx_properties['replicate'])
        else:
            replicate = None
            
        return generate_well_parameters(strain, od, source, src_well, replicate)


    def assign_replicate(self, strain, od, replicates):
        if strain not in replicates:
            replicates[strain] = {}
        strain_replicates = replicates[strain]

        if od not in strain_replicates:
            strain_replicates[od] = 0
        replicate = strain_replicates[od]
        strain_replicates[od] = strain_replicates[od] + 1
        return replicate
            


    
    def fulfill(self, logger, measurement_files, plate_id, overwrite_request=False):
        controls = { "A12" : {"strain": "WT-Live-Control"},
                     "B12" : {"strain": "WT-Dead-Control"},
                     "C12"  : {"strain": "NOR-00-Control"}}
        
        plate = self['plates'][plate_id]
        replicates = {}
            
        for aliquot, aliquot_properties in plate['wells'].items():
            laliquot = aliquot.lower()

            if aliquot in controls:
                continue
            
            if laliquot in measurement_files.keys():
                logger.info("handling " + str(laliquot))
                if overwrite_request:
                    plate['wells'][aliquot] = self.generate_well_parameters_from_tx(measurement_files['source_container'], laliquot, measurement_files[laliquot]['properties'])
                    aliquot_properties = plate['wells'][aliquot]
                    plate['source_container'] = measurement_files['source_container'].id
                    plate['source_container_aliquots'] = get_container_aliquots(measurement_files['source_container'].aliquots)
                logger.info("adding measurement: " + measurement_files[laliquot]['file'])
                logger.info("to " + laliquot)
                logger.info(measurement_files[laliquot])
                if 'measurements' in aliquot:
                    aliquot_properties['measurements'].append(measurement_files[laliquot]['file'])
                else:
                    aliquot_properties['measurements'] = [measurement_files[laliquot]['file']]
                logger.info(aliquot_properties)

                if aliquot_properties['strain'] is not None and aliquot_properties['od'] is not None:
                    aliquot_properties['replicate'] = self.assign_replicate(aliquot_properties['strain']['gate'], aliquot_properties['od'], replicates)
                else:
                    aliquot_properties['replicate'] = None
                
        ## Get control measurements
        logger.info("adding controls")
        for aliquot, control in controls.items():
            laliquot = aliquot.lower()
            control['measurements'] = [measurement_files[laliquot]['file']]
            control['od'] =  measurement_files[laliquot]['properties']['TargetOD']
            control['replicate'] = self.assign_replicate(control['strain'], control['od'], replicates)
        plate['wells'].update(controls)
        logger.info(controls)

        return self
                
            
            
    
class Sample(dict):
    """
    A desired sample for an experiment request.
    """

    def __init__(self):
        dict.__init__(self)

    def __str__(self):
        return json.dumps(self, indent=4, sort_keys=True, separators=(',', ': '))
