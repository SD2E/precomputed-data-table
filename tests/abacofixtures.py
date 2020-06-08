from __future__ import unicode_literals
import uuid
import pytest
from hashids import Hashids
HASH_SALT = 'gGvyjcUGJWNa2NxJ4aTF8n7yALYnZvs7'


def abaco_uuid():
    '''
    Generate an Abaco-style hashid
    '''
    hashids = Hashids(salt=HASH_SALT)
    _uuid = uuid.uuid1().int >> 64
    return hashids.encode(_uuid)

@pytest.fixture(scope='session')
def actor_id():
    return abaco_uuid()


@pytest.fixture(scope='session')
def exec_id():
    return abaco_uuid()
