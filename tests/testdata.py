__author__ = 'mwvaughn'

import json
import os
import sys
from past.builtins import basestring
from abacofixtures import abaco_uuid

HERE = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)


class TestData(object):
    '''Loads from tests/data/executions.json'''
    def __init__(self):

        self.dat = self.file_to_json('data/executions.json')
        for idx in range(0, len(self.dat)):
            for (k, v) in self.dat[idx].items():
                if isinstance(v, basestring):
                    # Randomizes the actor and exec identifiers
                    # so that outputs whose names incorporate
                    # those strings are unique to the session
                    if '!hashid' in v:
                        self.dat[idx][k] = abaco_uuid() + '.local'

    def file_to_json(self, filename):
        return json.load(open(os.path.join(HERE, filename)))

    def data(self, key=None):
        if key is None:
            return self.dat
        else:
            return self.dat.get(key, None)


class Secrets(object):
    '''Loads from the top-level secrets.json file'''
    def __init__(self):

        self.dat = self.file_to_json(os.path.join(PARENT, 'secrets.json'))

    def file_to_json(self, filename):
        fpath = os.path.join(CWD, filename)
        if os.path.isfile(fpath):
            return json.load(open(os.path.join(CWD, filename)))
        else:
            return {}

    def data(self, key=None):
        if key is None:
            return self.dat
        else:
            return self.dat.get(key, None)
