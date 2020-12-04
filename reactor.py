import os
import copy
import json
import jsonschema
import record_product_info as rpi

from datetime import datetime
from pymongo import MongoClient

from agavepy.agave import Agave
from attrdict import AttrDict
from datacatalog import mongo
from datacatalog import linkedstores
from datacatalog.managers.pipelinejobs import ReactorManagedPipelineJob as Job
from datacatalog.tokens import get_admin_token
from datacatalog.stores import StorageSystem
from reactors.runtime import Reactor, agaveutils
from requests.exceptions import HTTPError
from external_apps import perform_metrics as ea_pm

def join_posix_agave(joinList):
    new_path = '/'.join(arg.strip("/") for arg in joinList)
    new_path = os.path.join('/', new_path)
    return new_path

def get_database(r):
    mongodb = r.settings['mongodb']
    r.logger.debug("mongodb: {}".format(mongodb))
    mongodb_uri = mongo.get_mongo_uri(mongodb)
    r.logger.debug("mongodb_uri: {}".format(mongodb_uri))
    database_name = mongodb['database']
    r.logger.debug('database_name: {}'.format(database_name))
    myclient = MongoClient(mongodb_uri)
    return mongodb, myclient[database_name]

def load_structured_request(experiment_id, r):
    (mongodb, database) = get_database(r)
    ref_store = linkedstores.structured_request.StructuredRequestStore(mongodb) 

    query={}
    query['experiment_id'] = experiment_id
    matches = ref_store.query(query)
    
    # There should be at most one match
    if matches.count() == 0:
        return None
    else:
        return matches[0]

def aggregate_records(m, r):
    if "analysis" not in m:
        raise Exception("missing analysis")
    else:
        analysis = m.get("analysis")
    if "mtype" not in m:
        raise Exception("missing mtype")
    else:
        mtype = m.get("mtype")
    if "experiment_reference" not in m:
        raise Exception("missing experiment_reference")
    else:
        experiment_reference = m.get("experiment_reference")       
    if "data_converge_dir" not in m:
        raise Exception("missing data_converge_dir")
    else:
        data_converge_dir = m.get("data_converge_dir")
    if "parent_result_dir" not in m:
        raise Exception("missing parent_result_dir")
    else:
        parent_result_dir = m.get("parent_result_dir")
        
    (storage_system, dirpath, leafdir) = agaveutils.from_agave_uri(data_converge_dir)
    root_dir = StorageSystem(storage_system, agave=r.client).root_dir
    data_converge_dir2 = join_posix_agave([root_dir, dirpath, leafdir])
    r.logger.info("data_converge_dir2: {}".format(data_converge_dir2))
    
    (storage_system, dirpath, leafdir) = agaveutils.from_agave_uri(parent_result_dir)
    root_dir = StorageSystem(storage_system, agave=r.client).root_dir
    parent_result_dir2 = join_posix_agave([root_dir, dirpath, leafdir])
    r.logger.info("parent_result_dir2: {}".format(parent_result_dir2))

    analysis_dir = analysis + "__" + mtype if analysis == "perform-metrics" else analysis
    r.logger.info("analysis_dir: {}".format(analysis_dir))
    analysis_record_path = os.path.join(parent_result_dir2, analysis_dir, "record.json")
    if os.path.exists(analysis_record_path) is False:
        r.logger.info("{} doesn't exist".format(analysis_record_path))
        exit(2)

    record_path = os.path.join(parent_result_dir2, "record.json")
    # check for existing record.json
    if 'record.json' not in os.listdir(parent_result_dir2):
        record = rpi.make_product_record(experiment_reference, parent_result_dir2, data_converge_dir2)
        open(record_path, 'w+')
    else:
        with open(record_path, 'r') as jfile:
            record = json.load(jfile)

    with open(analysis_record_path, 'r') as analysis_json_file:
        analysis_record = json.load(analysis_json_file)
        if analysis_record:
            if analysis == "perform-metrics":
                analysis_label = analysis + "__" + mtype
                record["analyses"][analysis_label] = {}
                record["analyses"][analysis_label]["perform_metrics version"] = analysis_record["perform_metrics version"]
                record["analyses"][analysis_label]["files"] = []
                pm_record = {}
                pm_record["data_converge input"] = analysis_record["data_path"]
                pm_record["precomputed_data_table outputs"] = analysis_record["files"]
                record["analyses"][analysis_label]["files"].append(pm_record)
            else:
                record["analyses"][analysis] = analysis_record["analyses"][analysis]
            r.logger.info("updating {}".format(record_path))
            with open(record_path, 'w') as jfile:
                json.dump(record, jfile, indent=2)
            
def launch_omics(m, r):
    if "input_dir" not in m:
        raise Exception("missing input_dir")
    else:
        input_dir = m.get("input_dir")
            
    if "experiment_id" not in m:
        raise Exception("missing experiment_id")
    else:
        experiment_id = m.get("experiment_id")
        
    if "config_file" not in m:
        raise Exception("missing config_file")
    else:
        config_file = m.get("config_file")        
        
    analysis = m.get("analysis")
        
    input_counts_path = os.path.join(input_dir, experiment_id + "_ReadCountMatrix_preCAD_transposed.csv")


    sr = load_structured_request(experiment_id, r)
    experiment_ref = sr["experiment_reference"]
    state = "complete"
    
    app_job_def_inputs = {}
    app_job_def_inputs['input_data'] = agaveutils.to_agave_uri(systemId="data-sd2e-community", dirPath=input_counts_path)
    r.logger.info("input_data: {}".format(app_job_def_inputs['input_data']))
    
    job_data = copy.copy(m)
    archive_path = os.path.join(state, experiment_ref, analysis)
    r.logger.info("archive_path: {}".format(archive_path))
    product_patterns = []     

    job = Job(r,
              experiment_id=experiment_id,
              data=job_data,
              product_patterns=product_patterns,
              archive_system = 'data-sd2e-projects.sd2e-project-48',
              archive_path=archive_path)    
    job.setup()
    
    token_key = r.context["CATALOG_ADMIN_TOKEN_KEY"]
    atoken = get_admin_token(token_key)

    try:
        job.reset(token=atoken)
    except:
        job.ready(token=atoken)
            
    archive_path = job.archive_path
    r.logger.info("archive_path: {}".format(archive_path))
    r.logger.info('job.uuid: {}'.format(job.uuid))

    # maxRunTime should probably be determined based on experiment size        
    job_def = {
        "appId": r.settings.agave_omics_tools_app_id,
        "name": "precomputed-data-table-omics-tools" + r.nickname,
        "parameters": {"config_file": config_file},
        "maxRunTime": "48:00:00",
        "batchQueue": "all"
    }
    
    job_def["inputs"] = app_job_def_inputs

    # First, set the preferred archive destination and ensure the job archives
    job_def["archivePath"] = job.archive_path
    job_def["archiveSystem"] = "data-sd2e-projects.sd2e-project-48"
    job_def["archive"] = True
    job_def["notifications"] = [
            {
                "event": "RUNNING",
                "persistent": True,
                "url": job.callback + "&status=${JOB_STATUS}"
            }
        ]

    r.logger.info('Job Def: {}'.format(job_def))

    ag_job_id = None
    try:
        resp = r.client.jobs.submit(body=job_def)
        r.logger.debug("resp: {}".format(resp))
        if "id" in resp:
            ag_job_id = resp["id"]
            # Now, send a "run" event to the Job, including for the sake of
            # keeping good records, the Agave job ID.
            job.run({"launched": ag_job_id})

    except HTTPError as h:
        # Report what is likely to be an Agave-specific error
        raise Exception("Failed to submit job", h)

    except Exception as exc:
        # Report what is likely to be an error with this Reactor, the Data
        # Catalog, or the PipelineJobs system components
        raise Exception("Failed to launch {}".format(job.uuid), exc)

    # Optional: Send an 'update' event to the PipelineJob's
    # history commemorating a successful run for this Reactor.
    try:
        job.update({"note": "Reactor {} ran to completion".format(rx.uid)})
    except Exception:
        pass

    # I like to annotate the logs with a terminal success message
    r.on_success("Launched Agave job {} in {} usec".format(ag_job_id, r.elapsed()))    
    
def launch_app(m, r):
    
    analysis = m.get("analysis")
    
    if "experiment_ref" not in m:
        raise Exception("missing experiment_ref")
    else:
        experiment_ref = m.get("experiment_ref")
    if "data_converge_dir" not in m:
        raise Exception("missing data_converge_dir")
    else:
        data_converge_dir = m.get("data_converge_dir")
    if "datetime_stamp" not in m:
        raise Exception("missing datetime_stamp")
    else:
        datetime_stamp = m.get("datetime_stamp")

    state = "complete" if "complete" in data_converge_dir.lower() else "preview"
    
    r.logger.info("experiment_ref: {} data_converge_dir: {} analysis: {}".format(experiment_ref, data_converge_dir, analysis))
    (storage_system, dirpath, leafdir) = agaveutils.from_agave_uri(data_converge_dir)
    root_dir = StorageSystem(storage_system, agave=r.client).root_dir
    data_converge_dir2 = join_posix_agave([root_dir, dirpath, leafdir])
    r.logger.info("data_converge_dir2: {}".format(data_converge_dir2))
    
    # Capture initial parameterization passed in as message
    job_data = copy.copy(m)

    mongodb, database = get_database(r)
    experiment_ids = []
    experiments = database.structured_requests
    query = { "experiment_reference" : experiment_ref}
    matches = database.structured_requests.find(query)
    if matches.count() == 0:
        raise Exception("structured requests not found for {}".format(experiment_ref))
    elif matches.count() == 1:
        experiment_ids.append(matches[0]["experiment_id"])
    else:
        for match in matches:
            if match["derived_from"]:
                experiment_ids.append(match["experiment_id"])

    archive_path = os.path.join(state, experiment_ref, datetime_stamp, analysis)
    r.logger.info("archive_path: {}".format(archive_path))
        
    #meta_file_name = '__'.join([experiment_ref, 'fc_meta.csv'])
    #meta_with_absolute_path = os.path.join(data_converge_dir, meta_file_name)
    #if os.path.exists(meta_with_absolute_path):
    #    r.logger.info("meta_with_absolute_path: {}".format(meta_with_absolute_path))
    #    derived_using = [meta_with_absolute_path]
    #else:
    #    derived_using = []
    #r.logger.info("meta_with_absolute_path: {}".format(meta_with_absolute_path))

    mtype = None
    control_set_dir = None
    product_patterns = []
    if analysis == "perform-metrics":
        if "mtype" not in m:
            raise Exception("missing mtype")

        mtype = m.get("mtype")        
        output_path_parent = os.path.join(state, experiment_ref, datetime_stamp)
        job_def, product_patterns = ea_pm.get_job_template("data-sd2e-projects.sd2e-project-48", output_path_parent, data_converge_dir, experiment_ref, mtype)
    else:
        # maxRunTime should probably be determined based on experiment size  
        parameter_dict = {"experiment_ref": experiment_ref, "data_converge_dir": data_converge_dir2, "analysis": analysis}
        if control_set_dir:
            parameter_dict['control_set_dir'] = control_set_dir

        if analysis == "xplan-od-growth-analysis":
            app_id = r.settings.agave_growth_analysis_app_id
            parameter_dict["sift_ga_sbh_url"] = r.context["_REACTOR_SBH_URL"]
            parameter_dict["sift_ga_sbh_user"] = r.context["_REACTOR_SBH_USER"]
            parameter_dict["sift_ga_sbh_password"] = r.context["_REACTOR_SBH_PASSWORD"]
            parameter_dict["sift_ga_mongo_user"] = r.context["_REACTOR_CAT_DB_USER"]
            pr_file_name = '__'.join([experiment_ref, 'platereader.csv'])
            pr_file_path = os.path.join(data_converge_dir, pr_file_name)
            r.logger.info("pr_file_path: {}".format(pr_file_path))
            product_patterns = [
                {'patterns': ['.csv$'],
                 'derived_from': [pr_file_path],
                 'derived_using': []
                }]
        elif analysis == "fcs_signal_prediction":
            app_id = r.settings.agave_fcs_signal_prediction_app_id
            fc_file_name = '__'.join([experiment_ref, 'fc_raw_events.json'])
            fc_file_path = os.path.join(data_converge_dir, fc_file_name)
            r.logger.info("fc_file_path: {}".format(fc_file_path))
            product_patterns = [
                {'patterns': ['.csv$'],
                 'derived_from': [fc_file_path],
                 'derived_using': []
                }]
        elif analysis == "live-dead-prediction":
            if "control_set_dir" not in m:
                raise Exception("missing control_set_dir")

            control_set_dir0 = m.get("control_set_dir")
            (storage_system, dirpath, leafdir) = agaveutils.from_agave_uri(control_set_dir0)
            root_dir = StorageSystem(storage_system, agave=r.client).root_dir
            control_set_dir = join_posix_agave([root_dir, dirpath, leafdir])
            r.logger.info("control_set_dir: {}".format(control_set_dir))
              
            app_id = r.settings.agave_live_dead_prediction_app_id
            fc_file_name = '__'.join([experiment_ref, 'fc_raw_events.json'])
            fc_file_path = os.path.join(data_converge_dir, fc_file_name)
            r.logger.info("fc_file_path: {}".format(fc_file_path))
            product_patterns = [
                {'patterns': ['.csv$'],
                 'derived_from': [fc_file_path],
                 'derived_using': []
                }]
        elif analysis == "wasserstein":
            app_id = r.settings.agave_wasserstein_app_id
            fc_raw_log10_stats_file_name = '__'.join([experiment_ref, 'fc_raw_log10_stats'])
            fc_raw_log10_stats_file_path = os.path.join(data_converge_dir, fc_raw_log10_stats_file_name + '.csv')
            r.logger.info("fc_raw_log10_stats_file_path: {}".format(fc_raw_log10_stats_file_path))
            fc_etl_stats_file_name = '__'.join([experiment_ref, 'fc_etl_stats'])
            fc_etl_stats_file_path = os.path.join(data_converge_dir, fc_etl_stats_file_name + '.csv')
            r.logger.info("fc_etl_stats_file_path: {}".format(fc_etl_stats_file_path))
            product_patterns = [
                {'patterns': ['^.*fc_raw_log10_stats.*\.(csv|json)$'],
                 'derived_from': [fc_raw_log10_stats_file_path],
                 'derived_using': []
                },
                {'patterns': ['^.*fc_etl_stats.*\.(csv|json)$'],
                 'derived_from': [fc_etl_stats_file_path],
                 'derived_using': []
                }
            ]

        job_def = {
            "appId": app_id,
            "name": "precomputed-data-table-app" + r.nickname,
            "parameters": parameter_dict,
            "maxRunTime": "8:00:00",
            "batchQueue": "all"
        }
    
        # First, set the preferred archive destination and ensure the job archives
        job_def["archivePath"] = archive_path
        job_def["archiveSystem"] = "data-sd2e-projects.sd2e-project-48"
        job_def["archive"] = True

    r.logger.debug("Instantiating job with product_patterns: {}".format(product_patterns))

    job = Job(r,
              experiment_id=experiment_ids,
              data=job_data,
              product_patterns=product_patterns,
              archive_system = job_def["archiveSystem"],
              archive_path=job_def["archivePath"])

    job.setup()

    token_key = r.context["CATALOG_ADMIN_TOKEN_KEY"]
    atoken = get_admin_token(token_key)

    try:
        job.reset(token=atoken)
    except:
        job.ready(token=atoken)
    
    archive_path = job.archive_path
    r.logger.info("archive_path: {}".format(archive_path))
    r.logger.info('job.uuid: {}'.format(job.uuid))

    job_def["notifications"] = [
            {
                "event": "RUNNING",
                "persistent": True,
                "url": job.callback + "&status=${JOB_STATUS}"
            }
        ]

    r.logger.info('Job Def: {}'.format(job_def))

    ag_job_id = None
    try:
        r.logger.info("submit Tapis job")
        resp = r.client.jobs.submit(body=job_def)
        r.logger.info("resp: {}".format(resp))
        if "id" in resp:
            ag_job_id = resp["id"]
            # Now, send a "run" event to the Job, including for the sake of
            # keeping good records, the Agave job ID.
            job.run({"launched": ag_job_id})

    except HTTPError as h:
        # Report what is likely to be an Agave-specific error
        raise Exception("Failed to submit job", h)

    except Exception as exc:
        # Report what is likely to be an error with this Reactor, the Data
        # Catalog, or the PipelineJobs system components
        raise Exception("Failed to launch {}".format(job.uuid), exc)

    # Optional: Send an 'update' event to the PipelineJob's
    # history commemorating a successful run for this Reactor.
    try:
        job.update({"note": "Reactor {} ran to completion".format(rx.uid)})
    except Exception:
        pass

    # I like to annotate the logs with a terminal success message
    r.on_success("Launched Agave job {} in {} usec".format(ag_job_id, r.elapsed()))
    
def main():

    r = Reactor()
    m = r.context.message_dict
    r.logger.info("message: {}".format(m))
    r.logger.info("raw message: {}".format(r.context.raw_message))
    
    if "aggregate_records" in m:
        aggregate_records(m, r)
    else:
        if "analysis" not in m:
            raise Exception("missing analysis")
        else:
            analysis = m.get("analysis")
    
        if analysis == "omics_tools":
            launch_omics(m, r)
        else:
            launch_app(m, r)

if __name__ == '__main__':
    main()