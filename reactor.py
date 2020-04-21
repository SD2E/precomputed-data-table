import os

import json
import jsonschema

from agavepy.agave import Agave
from attrdict import AttrDict
from datacatalog.managers.pipelinejobs import ReactorManagedPipelineJob as Job
from datacatalog.tokens import get_admin_token
from reactors.runtime import Reactor, agaveutils
from requests.exceptions import HTTPError
from pysd2cat.data import pipeline

class formatChecker(jsonschema.FormatChecker):
    def __init__(self):
        jsonschema.FormatChecker.__init__(self)

def main():

    r = Reactor()
    m = r.context.message_dict
    
    try:
        ag = Agave(api_server=os.environ.get('_abaco_api_server'),
                   token=os.environ.get('agave_token'))

        resp = ag.jobs.list()
        r.logger.info('Custom Agave Client Resp: {}'.format(resp))
    except Exception as exc:
        r.logger.error('Error: {}'.format(exc))

    for var in os.environ:
        r.logger.info('os.environ.{}: {}'.format(var, os.environ[var]))

    r.logger.info('Abaco Access Token: {}'.format(os.environ['_abaco_access_token']))
    r.logger.info('Client Token: {}'.format(r.client._token))
        
    r.logger.info("message: {}".format(m))
    experiment_id = m.get("experiment_id", None)
    r.logger.info("experiment_id: {}".format(experiment_id))
    sample_ids = []
    samples = pipeline.get_experiment_samples(experiment_id)
    for sample in samples:
        sample_ids.append(sample['sample_id'])
    product_patterns = [
        {'patterns': ['samples_labelled.csv'],
         'derived_from': sample_ids,
         'derived_using': []
        }] 

    r.logger.debug("Instantiating job")

    job = Job(r, data=m.get("data", {}),
#              session=r.session,
              experiment_id=experiment_id,
              product_patterns=product_patterns)

    # At this point, there is an entry in the MongoDB.jobs collection, the
    # job UUID has been assigned, archive_path has been set, and the contents
    # of 'data' passed at init() and/or setup() are merged and stored in
    # the 'data' field of the job.
    #
#    token = get_admin_token(r.settings.admin_token_key)
#    try:
#        job.reset(token=token)
        # job.ready(token=token)
#    except Exception as exc:
#        r.logger.warning('Reset failed: {}'.format(exc))
    
    job.setup()
    r.logger.info("PipelineJob.uuid: {}".format(job.uuid))
    
    archive_path = job.archive_path
    r.logger.debug("archive_path: {}".format(archive_path))
    
    # At this point, but no later in the job lifecycle, the job can be erased
    # with job.cancel(). If there is a need to denote failure after the job has
    # started running, that should be achieved via job.fail().

    # Launching an Agave job from within a Reactor is well-documented
    # elsewhere. This example extends the common use base by configuring the
    # Agave job to POST data to a HTTP URL when it reaches specific stages in
    # its lifecycle. These POSTS can be configured and that is leveraged to
    # notify the PipelineJobs system as to the status of the Agave job. The
    # example shown here is a generalized solution and can be used by any
    # Agave job to communicate with an Abaco Reactor.
    #
    # Another key difference between this example and the common use case is
    # that the job's archivePath (i.e. the final destination for job output
    # files) is explicitly set. Specifically, it is set to a path that is
    # managed by the ManagedPipelineJob class.
        
    job_def = {
        "appId": r.settings.agave_app_id,
        "name": "sample-quality-check-" + r.nickname,
        "parameters": {"experiment_id": experiment_id},
        "maxRunTime": "00:15:00",
    }

    # First, set the preferred archive destination and ensure the job archives
    job_def["archivePath"] = job.archive_path
    job_def["archiveSystem"] = job.archive_system
    job_def["archive"] = True

    # Second, add event notifications to the job definition
    #
    # Agave has many job statuses, and the PipelineJobs System
    # has mappings for all of them. The most important, which are
    # demonstrated below, are RUNNING, ARCHIVING_FINISHED, and FAILED.
    # These correspond to their analagous PipelineJobs states. This example
    # leverages ManagedPipelineJob's built-in method for getting a minimal
    # set of notifications for RUNNING, FINISHED, and FAILED job events.

    job_def["notifications"] = job.agave_notifications()

    r.logger.info('Job Def: {}'.format(job_def))
        
    # Submit the Agave job: The Agave job will send event its updates to
    # our example job via the Jobs Manager Reactor, This will take place even
    # after the execution of this Reactor has concluded. This is a good example
    # of asynchronous, event-driven programming. This is a remarkably scalabe,
    # and resilient approach, and its innate composability suggests creation of
    # complex, internlinked workflows.
    ag_job_id = None
    try:
        resp = r.client.jobs.submit(body=job_def)
        r.logger.debug("resp: {}".format(resp))
        if "id" in resp:
            ag_job_id = resp["id"]
            # Now, send a "run" event to the Job, including for the sake of
            # keeping good records, the Agave job ID.
            job.run({"launched": ag_job_id})
#        else:
            # Fail the PipelineJob if Agave job fails to launch
#            job.cancel()

    except HTTPError as h:
        # Report what is likely to be an Agave-specific error
        #http_err_resp = agaveutils.process_agave_httperror(h)
        #job.cancel({"cause": str(http_err_resp)})
        raise Exception("Failed to submit job", h)

    except Exception as exc:
        # Report what is likely to be an error with this Reactor, the Data
        # Catalog, or the PipelineJobs system components
        #job.cancel()
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