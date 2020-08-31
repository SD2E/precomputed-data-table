import transcriptic 
from transcriptic import commands, Container
from transcriptic.jupyter import objects
import hashlib
import csv
import json
import requests
import os
import zipfile
import shutil
import fnmatch
import glob

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)

def get_plate_well_properties(run_obj, container_name="flow_plate \(sytox\)"):
    #print("getting flow well properties")
    try:
        cs = run_obj.containers
    except Exception as e:
        cs = run_obj.containers

    l.debug("Getting container well properties: %s", container_name)
    l.debug("From: %s", str(cs))
    l.debug("Matches %s", str(cs.loc[cs['Name'].str.contains(container_name)]))
    flow_plate_id = cs.loc[cs['Name'].str.contains(container_name)].iloc[0]['ContainerId']
    l.debug("Have flow_plate_id: %s", flow_plate_id)
    flow_plate = transcriptic.container(flow_plate_id)
    aliquots = flow_plate.attributes['aliquots']
    properties = {}
    for a in flow_plate.attributes['aliquots']:
        #print(a)
        well = flow_plate.container_type.humanize(a['well_idx'])
        properties[well] = a['properties']
    l.debug("Have properties %s", str(properties))
    return properties

def _fcs_file_well_id(filename):
    #print("Getting well from filename: " + filename)
    return filename.split('_')[-1].split('.')[0].lower()


def get_source_container(run_obj):
    """
    Get the stock plate container that started the run, which preceded run_obj.
    """
    #print("Getting source container.")
    try:
        cs = run_obj.containers
    except Exception as e:
        cs = run_obj.containers
    source_plate_id = cs.loc[ cs['Name'].str.contains('YeastGatesStockPlate')].iloc[0]['ContainerId']
    return transcriptic.container(source_plate_id)


def _file_checksum(filePath):
    hash_sha = hashlib.sha1()
    with open(filePath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()

def get_fcs_files(run_obj, tx_email, tx_token,
                  work_dir, download_zip=True, logger=l,
                  container_name="flow_plate \(sytox\)",
                  data_set_name=None,
                  source_container_run=None):
    d = run_obj.data
    if data_set_name:
        d = d.loc[d['Name'].str.contains(data_set_name)]
    logger.info("Datasets: %s", str(d))
    for dataset in d['Datasets']:
        if dataset.data_type == "file":
            unzip_path = os.path.join(work_dir, 'fcs')
                
            if download_zip:
                logger.info(dataset.attributes["attachments"])
                key = dataset.attributes["attachments"][0]['key']
                headers = {'X-User-Email': tx_email, 'X-User-Token': tx_token}
                response = requests.get("https://secure.transcriptic.com/upload/url_for?key="+ key, headers=headers)
                logger.info("Requesting Data from: " + "https://secure.transcriptic.com/upload/url_for?key="+ key)
                zip_path = os.path.join(work_dir, 'fcs.zip')
                unzip_tmp_path = os.path.join(work_dir, 'fcs_tmp')
                with open(zip_path, 'wb') as f:
                    logger.info("Writing data to: " + zip_path)
                    #logger.info("Response: " + str(response))
                    f.write(response.content)

                # Unzip to tmp_path. Unzips to a directory a weird name
                zip_ref = zipfile.ZipFile(zip_path, 'r')
                zip_ref.extractall(unzip_tmp_path)
                zip_ref.close()

                # Search for the fcs files
                matches = []
                for root, dirnames, filenames in os.walk(unzip_tmp_path):
                    for filename in fnmatch.filter(filenames, '*.fcs'):
                        matches.append((os.path.join(root, filename), filename))

                if not os.path.exists(unzip_path):
                    os.makedirs(unzip_path)

                # Move all fcs files to a directory with a known name
                for full_path, filename in matches:
                    shutil.move(full_path, os.path.join(unzip_path, filename))

                shutil.rmtree(unzip_tmp_path)
                os.remove(zip_path)

            properties = get_plate_well_properties(run_obj, container_name=container_name)
            logger.debug(properties)
            # well id -> file info
            
            aliquots = {_fcs_file_well_id(f) : {'file': f,
                                'properties' : properties[_fcs_file_well_id(f).upper()],
                                'checksum': _file_checksum(f)}
                     for f in glob.glob(os.path.join(unzip_path, '*.fcs'))
                     if  _fcs_file_well_id(f).upper() in properties }
            files = { "aliquots" : aliquots }
            if source_container_run:
                files['source_container'] = get_source_container(source_container_run)
            logger.debug(files)
            return files

def get_tx_run(run_id):
    try:
        # In the reactor, the connection will already have been
        # made at this point. Don't worry if this fails. If the
        # connection failed up above, the Run instantion will
        # still cause an exception.
        connect()
    except Exception:
        pass
    return objects.Run(run_id)


def create_fcs_manifest_and_get_files(run_id, 
                                      transcriptic_email,
                                      transcriptic_token,
                                      fcs_path='.',
                                      download_zip=True,
                                      logger=l,
                                      container_name="flow_plate \(sytox\)",
                                      data_set_name=None,
                                      source_container_run=None):
   
    run_obj = get_tx_run(run_id)
    src_run_obj = None
    if source_container_run:
        src_run_obj = get_tx_run(source_container_run)
        
    fcs_files = get_fcs_files(run_obj,
                              transcriptic_email,
                              transcriptic_token,
                              fcs_path,
                              download_zip=download_zip,
                              logger=logger,
                              container_name=container_name,
                              source_container_run=src_run_obj,
                              data_set_name=data_set_name)
    return fcs_files
    #create_fcs_manifest(run_obj, experiment_id, fcs_files, manifest_path)
    #print("Have FCS manifest")

def get_transcriptic_api(settings):
    """Connect (without validation) to Transcriptic.com"""
    try:
        return transcriptic.Connection(**settings)
    except Exception:
        raise
