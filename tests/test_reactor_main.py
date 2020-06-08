from __future__ import unicode_literals
import json
import os
import sys
import semver
import yaml

# Import various types we will assert against
from attrdict import AttrDict
# from agavepy.agave import Agave
# from builtins import str
# from logging import Logger

CWD = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)
sys.path.insert(0, '/')

from reactors.runtime import Reactor

import pytest
from agavefixtures import credentials, agave
import testdata

# Import the 'reactor.py' file.
# TODO - Move this to individual tests to enable parameterization
import reactor


@pytest.fixture(scope='session')
def test_data():
    return testdata.TestData().data()


@pytest.fixture(scope='session')
def secrets_data():
    return testdata.Secrets().data()


def test_imports_sbol_version():
    '''Ensure sbol imports and is a recent version'''
    import sbol
    version_info = semver.parse_version_info(sbol.__version__)
    assert version_info > (2, 3, 0)


def test_imports_synbiohub_adapter():
    '''SynbioHub can be imported'''
    from synbiohub_adapter.upload_sbol import SynBioHub, \
        BadLabParameterError, UndefinedURIError


def test_imports_reactor_py():
    '''File reactor.py can be imported'''
    import reactor
    assert 'main' in dir(reactor)


def test_config_yml():
    '''File config.yml is loadable'''
    with open(os.path.join(CWD, 'config.yml'), "r") as conf:
        y = yaml.safe_load(conf)
        assert isinstance(y, dict)


def test_config_agave_client(agave, credentials):
    '''Able to instantiate an Agave client'''
    p = agave.profiles.get()
    assert p['username'] == credentials['username']


def test_test_data(monkeypatch, test_data, secrets_data):
    '''Ensure test data loads OK'''
    execution = test_data
    assert isinstance(execution, list)
    # executions.json contains an array of dicts with env variables to set
    for k in execution[0].keys():
        monkeypatch.setenv(k, execution[0].get(k, ""))
    assert os.environ.get('_abaco_actor_id', None) is not '!hashid'
    for s in secrets_data.keys():
        monkeypatch.setenv(s, secrets_data[s])
    assert os.environ.get('_REACTOR_SLACK_WEBHOOK', None) is not None
    assert os.environ.get('_REACTOR_LOGS_TOKEN', None) is not None


def test_reactor_init():
    '''Ensure Reactor object can initialize'''
    r = reactor.Reactor()
    assert isinstance(r, Reactor)


def test_reactor_read_config():
    '''Validate config.yml loads config.yml properly'''
    r = reactor.Reactor()
    assert isinstance(r.settings, AttrDict)
    # it doesn't matter what keys one puts here - the idea is to ensure
    # that the config.yml is valid YAML and thus loadable as a dict
    assert 'logs' in r.settings
    assert r.settings.logs.level is not None

#def test_fn_resilient_files_get():
#    import reactor as r
#    reactor = r.Reactor()
#    download = r.resilient_files_get(agaveClient=reactor.client,
#                                     agaveAbsolutePath='/biofab/yeast-gates/aq_10545/3/manifest/manifest.json',
#                                     systemId='data-sd2e-community',
#                                     localFilename='test_manifest.json')

    assert os.path.basename(download) == 'test_manifest.json'


# @pytest.mark.skip(reason="not implemented")
# def test_fn_resilient_files_list():
#     import reactor as r
#     reactor = r.Reactor()
#     listing = r.resilient_files_list(agaveClient=reactor.client,
#                                      agaveAbsolutePath='/biofab/yeast-gates/aq_10545/3/manifest/manifest.json',
#                                      systemId='data-sd2e-community')

#     assert listing is True


# def test_reactor_main(monkeypatch,
#                       test_data, secrets_data):
#     '''emulate an execution directly from contents of executions.json'''
#     execution = test_data
#     for k in execution[0].keys():
#         monkeypatch.setenv(k, execution[0].get(k, ""))
#     for s in secrets_data.keys():
#         monkeypatch.setenv(s, secrets_data[s])
#     # s/reactor.py/reactor/
#     import reactor as r
#     with pytest.raises(Exception) as pytest_wrapped_e:
#         r.main()
#     assert pytest_wrapped_e.type == SystemExit
#     assert pytest_wrapped_e.value.code == 0
