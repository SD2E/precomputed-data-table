import os
import json
from jsonschema import validate
from jsonschema import ValidationError

SCHEMA = '/../message.jsonschema'
HERE = os.path.dirname(os.path.abspath(__file__))
MESSAGES = os.path.join(HERE, 'data', 'messages.json')


def test_validate_message_schema():
    '''JSON schema validator works with message schema and JSON data'''

    with open(MESSAGES) as messages:
        messages_data = json.loads(messages.read())

    valid_count = 0
    for msg in messages_data:

        object_json = msg.get("object")
        expected_result = msg.get("valid")

        with open(SCHEMA) as schema_file:
            schema_json = json.loads(schema_file.read())
            validated = None
            try:
                validate(object_json, schema_json)
                validated = True
            except ValidationError as err:
                validated = False

            if validated == expected_result:
                valid_count = valid_count + 1
            else:
                print("MISMATCH: {}".format(object_json))

    assert valid_count == len(messages_data), \
        "Mismatch in validation expectation vs observed."
