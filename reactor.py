import os
import copy
import json
import jsonschema

from agavepy.agave import Agave
from attrdict import AttrDict
from datacatalog import mongo
from datacatalog.managers.pipelinejobs import ReactorManagedPipelineJob as Job
from datacatalog.tokens import get_admin_token
from datacatalog.stores import StorageSystem
from reactors.runtime import Reactor, agaveutils
from requests.exceptions import HTTPError
from datetime import datetime

from pymongo import MongoClient

def join_posix_agave(joinList):
    new_path = '/'.join(arg.strip("/") for arg in joinList)
    new_path = os.path.join('/', new_path)
    return new_path

def get_mongodb_database(r):
    mongodb = r.settings['mongodb']
    r.logger.debug("mongodb: {}".format(mongodb))
    mongodb_uri = mongo.get_mongo_uri(mongodb)
    r.logger.debug("mongodb_uri: {}".format(mongodb_uri))
    database_name = mongodb['database']
    r.logger.debug('database_name: {}'.format(database_name))
    myclient = MongoClient(mongodb_uri)
    database = myclient[database_name]
    
    return database

def main():

    r = Reactor()
    m = r.context.message_dict
    r.logger.info("message: {}".format(m))
    r.logger.info("raw message: {}".format(r.context.raw_message))

    if "analysis" not in m:
        raise Exception("missing analysis")
    else:
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

    database = get_mongodb_database(r)
    experiment_ids = []
    experiments = database.structured_requests
    query = { "experiment_reference" : experiment_ref}
    matches = database.structured_requests.find(query)
    if matches.count() == 0:
        raise Exception("structured requests not found for {}".format(experiment_ref))
    elif matches.count() == 1:
        experiment_ids.append(matches[0]["experiment_id"])
    else:
        for m in matches:
            if m["derived_from"]:
                experiment_ids.append(m["experiment_id"])
    
    pr_file_name = '__'.join([experiment_ref, 'platereader.csv'])
    pr_with_absolute_path = os.path.join(data_converge_dir, pr_file_name)
    r.logger.info("pr_with_absolute_path: {}".format(pr_with_absolute_path))
    meta_file_name = '__'.join([experiment_ref, 'fc_meta.csv'])
    meta_with_absolute_path = os.path.join(data_converge_dir, meta_file_name)
    if os.path.exists(meta_with_absolute_path):
        r.logger.info("meta_with_absolute_path: {}".format(meta_with_absolute_path))
        derived_using = [meta_with_absolute_path]
    else:
        derived_using = []
    r.logger.info("meta_with_absolute_path: {}".format(meta_with_absolute_path))
    product_patterns = [
        {'patterns': ['.csv$'],
         'derived_from': [pr_with_absolute_path],
         'derived_using': derived_using
        }] 

    r.logger.debug("Instantiating job with product_patterns: {}".format(product_patterns))

    #job_data["datetime_stamp"] = datetime_stamp
    archive_path = os.path.join(state, experiment_ref, datetime_stamp, analysis)
    
    r.logger.debug("archive_path: {}".format(archive_path))
    
    job = Job(r,
              experiment_id=experiment_ids,
              data=job_data,
              product_patterns=product_patterns,
              archive_system = 'data-sd2e-projects.sd2e-project-48',
              archive_path=archive_path)

    job.setup()

    #token_key = r.context["CATALOG_ADMIN_TOKEN_KEY"]
    #atoken = get_admin_token(token_key)

    #try:
    #    job.reset(token=atoken)
    #except:
    #    job.ready(token=atoken)
    
    archive_path = job.archive_path
    r.logger.info("archive_path: {}".format(archive_path))
    r.logger.info('job.uuid: {}'.format(job.uuid))

    # maxRunTime should probably be determined based on experiment size        
    job_def = {
        "appId": r.settings.agave_app_id,
        "name": "precomputed-data-table-app" + r.nickname,
        "parameters": {"experiment_ref": experiment_ref, "data_converge_dir": data_converge_dir2, "analysis": analysis},
        "maxRunTime": "24:00:00",
        "batchQueue": "long"
    }

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

if __name__ == '__main__':
    main()