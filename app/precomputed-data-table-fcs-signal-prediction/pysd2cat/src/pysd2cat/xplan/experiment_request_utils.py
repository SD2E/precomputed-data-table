def get_source_for_strain(aliquots, strain):
    for i in aliquots.index:
        if aliquot_matches(aliquots[i:i+1], strain):
            alq = aliquots[i:i+1].to_dict()
            alq['Volume'] = str(alq['Volume']) # Volume is Unit class
            alq['index'] = i
            return alq
    raise Exception("Could not find aliquot for strain: " + str(strain))

def aliquot_matches(aliquot, strain):
    if 'gate' in strain:
        #print(str(aliquot.iloc[0]['SynBioHub URI']))
        #print(strain['gate'])
        # FIXME add back following line after we get some more containers w/ URIs
        return strain['gate']['value'] == str(aliquot.iloc[0]['SynBioHub URI'])
        #return True
    else:
        raise Exception("Do not support WT assignment to destination plate yet!")
        # is a wild type
        return False

def generate_well_parameters(strain, od, source, src_well, replicate):
    well_parameters = {}
    well_parameters['strain'] = strain
    well_parameters['od'] = od
    well_parameters['source'] = source
    well_parameters['src_well'] = src_well
    well_parameters['replicate'] = replicate
    return well_parameters

def get_container_aliquots(aliquots):
        aliquot_info = {}
        for index, aliquot in aliquots.iterrows():
            key = str(index)
#            print(aliquot)
            name = str(aliquot['Name'])
            volume = str(aliquot['Volume'])
            properties = {}
            for prop in ['Gate', 'replicate', 'Circuit']:
                if prop in aliquot:
                    p = str(aliquot[prop])
                else:
                    p = None
                properties.update({prop:p})
                
            value = { "name" : name,
                      "volume" : volume,
                      "properties" : properties
                      }
            aliquot_info.update({key:value})
        return aliquot_info

def gate_info(colony_data):
    """
    Get the base id for a colony URI.
    Need to specify gate here to preserve existing functionality.
    """
    mapping = {
               'UWBF_7376': {'base_id': 'UWBF_AND', 'input': [0, 0]},
               'UWBF_7375': {'base_id': 'UWBF_AND', 'input': [0, 1]},
               'UWBF_7373': {'base_id': 'UWBF_AND', 'input': [1, 0]},
               'UWBF_7374': {'base_id': 'UWBF_AND', 'input': [1, 1]},

               'UWBF_8544': {'base_id': 'UWBF_NAND', 'input': [0, 0]},
               'UWBF_8545': {'base_id': 'UWBF_NAND', 'input': [0, 1]},
               'UWBF_8543': {'base_id': 'UWBF_NAND', 'input': [1, 0]},
               'UWBF_8542': {'base_id': 'UWBF_NAND', 'input': [1, 1]},

               'UWBF_6390': {'base_id': 'UWBF_NOR', 'input': [0, 0]},
               'UWBF_6388': {'base_id': 'UWBF_NOR', 'input': [0, 1]},
               'UWBF_6389': {'base_id': 'UWBF_NOR', 'input': [1, 0]},
               'UWBF_6391': {'base_id': 'UWBF_NOR', 'input': [1, 1]},

               'UWBF_8225': {'base_id': 'UWBF_OR', 'input': [0, 0]},
               'UWBF_5993': {'base_id': 'UWBF_OR', 'input': [0, 1]},
               'UWBF_5783': {'base_id': 'UWBF_OR', 'input': [1, 0]},
               'UWBF_5992': {'base_id': 'UWBF_OR', 'input': [1, 1]},
               
               'UWBF_7300': {'base_id': 'UWBF_XNOR', 'input': [0, 0]},
               'UWBF_8231': {'base_id': 'UWBF_XNOR', 'input': [0, 1]},
               'UWBF_7377': {'base_id': 'UWBF_XNOR', 'input': [1, 0]},
               'UWBF_7299': {'base_id': 'UWBF_XNOR', 'input': [1, 1]},

               'UWBF_16970': {'base_id': 'UWBF_XOR', 'input': [0, 0]},
               'UWBF_16969': {'base_id': 'UWBF_XOR', 'input': [0, 1]},
               'UWBF_16968': {'base_id': 'UWBF_XOR', 'input': [1, 0]},
               'UWBF_16967': {'base_id': 'UWBF_XOR', 'input': [1, 1]},
               'UWBIOFAB_22544': {'base_id': 'UWBF_WT', 'input': None}}

    for chunk in mapping.keys():
        if chunk in colony_data['gate']:
            return mapping[chunk]

    raise Exception('Could not find colony base ID for {}'.format(uri))
    
def get_role(circuit_id):
    id_to_role = {
        'AND' :  'http://www.openmath.org/cd/logic1#and',
        'OR' : 'http://www.openmath.org/cd/logic1#or',
        'NAND' : 'http://www.openmath.org/cd/logic1#nand',
        'NOR' : 'http://www.openmath.org/cd/logic1#nor',
        'XOR' : 'http://www.openmath.org/cd/logic1#xor',
        'XNOR' : 'http://www.openmath.org/cd/logic1#xnor'
    }
    return id_to_role[circuit_id]

