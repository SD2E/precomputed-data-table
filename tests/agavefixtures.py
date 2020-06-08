from __future__ import unicode_literals
import json
import os
import sys
import pytest

import agavepy.agave as a

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.append('/')

# Valid environment variable prefixes to parameterize tests
PREFIXES = ['AGAVE', '_AGAVE', 'TACC', '_TACC']


@pytest.fixture(scope='session')
def credentials():
    '''
    Import Agave credentials for a testing session

    Order: user credential store, test file, environment
    '''
    credentials = {}
    # credential store
    if os.environ.get('AGAVE_CACHE_DIR', None) is not None:
        ag_cred_store = os.path.join(
            os.environ.get('AGAVE_CACHE_DIR'), 'current')
    else:
        ag_cred_store = os.path.expanduser('~/.agave/current')

    if os.path.exists(ag_cred_store):
        print("\nLoading credentials from the credential store: {}".format(ag_cred_store))
        tempcred = json.load(open(ag_cred_store, 'r'))
        credentials['apiserver'] = tempcred.get('baseurl', None)
        credentials['username'] = tempcred.get('username', None)
        credentials['password'] = tempcred.get('password', None)
        credentials['apikey'] = tempcred.get('apikey', None)
        credentials['apisecret'] = tempcred.get('apisecret', None)
        credentials['token'] = tempcred.get('access_token', None)
        credentials['refresh_token'] = tempcred.get('refresh_token', None)
        credentials['verify_certs'] = True
        credentials['client_name'] = tempcred.get('client_name', None)
        credentials['tenantid'] = tempcred.get('tenantid', None)
    # test file
    credentials_file = os.environ.get('creds', 'test_credentials.json')
    if os.path.exists(credentials_file):
        print(("\nLoading credentials from file: {}".format(credentials_file)))
        credentials = json.load(open(
            os.path.join(HERE, credentials_file), 'r'))
    # environment
    for env in ('apikey', 'apisecret', 'username', 'password',
                'apiserver', 'verify_certs', 'refresh_token',
                'token', 'client_name', 'tenantid'):
        for varname_root in PREFIXES:
            varname = varname_root + env.upper()
            if os.environ.get(varname, None) is not None:
                print("\nLoading credentials from environment")
                credentials[env] = os.environ.get(varname)
                break

    return credentials


@pytest.fixture(scope='session')
def agave(credentials):
    '''Return a functional Agave client'''
    aga = a.Agave(username=credentials.get('username'),
                  password=credentials.get('password'),
                  api_server=credentials.get('apiserver'),
                  api_key=credentials.get('apikey'),
                  api_secret=credentials.get('apisecret'),
                  token=credentials.get('token'),
                  refresh_token=credentials.get('refresh_token'),
                  verify=True)
    return aga
