"""
gather record product info and make data files summaries for this run of precomputed data table

:authors: robert c. moseley (robert.moseley@duke.edu) and  anastasia deckard (anastasia.deckard@geomdata.com)
"""

import hashlib
import os
import json
import pandas as pd

from datetime import datetime
import pymongo


def get_db_conn():
    dbURI = 'mongodb://readonly:WNCPXXd8ccpjs73zxyBV@catalog.sd2e.org:27020/admin?readPreference=primary'
    client = pymongo.MongoClient(dbURI)
    db = client.catalog_staging

    return db