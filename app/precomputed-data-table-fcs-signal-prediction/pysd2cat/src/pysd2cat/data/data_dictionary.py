from sbol import *
from synbiohub_adapter.query_synbiohub import *
from synbiohub_adapter.SynBioHubUtil import *
from synbiohub_adapter.SynBioHubUtil import SD2Constants

def get_sbh_query():
    sbh_query = SynBioHubQuery(SD2Constants.SD2_SERVER)
    sbh_query.login("sd2e", "jWJ1yztJl2f7RaePHMtXmxBBHwNt")
    return sbh_query

def get_uri_handle(uri):
    sbh_query = get_sbh_query()
    # Are we parsing a URI directly, not an internal lab id?
    query ="""PREFIX sbol: <http://sbols.org/v2#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX sd2: <http://sd2e.org#>
    SELECT ?id WHERE {{
        <https://hub.sd2e.org/user/sd2e/design/design_collection/1> sbol:member ?identity .
        ?identity <http://sd2e.org#Transcriptic_UID> ?id .
        VALUES (?identity) {{ (<{}>) }}
    }}""".format(uri)

    designs = sbh_query.fetch_SPARQL(SD2Constants.SD2_SERVER, query)

    # format return
    designs = sbh_query.format_query_result(designs, ['id'])
    return designs[0]

