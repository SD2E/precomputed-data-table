# use string constants for variables that have to be entered in code
# this makes sure we're not messing up by entering the wrong case or spelling
# it also consolidates 'complete lists',

class Names:

    CHALLENGE_PROBLEM = 'challenge_problem'
    YEAST_STATES = 'YEAST_STATES'

    #Strains
    STRAIN = 'strain'
    WT_DEAD_CONTROL='WT-Dead-Control'
    WT_LIVE_CONTROL = 'WT-Live-Control'
    NOR_00_CONTROL = "NOR-00-Control"
    STRAIN_NAME = 'strain_name'

    TIME = "Time"
    
    #Calibration
    LUDOX = 'LUDOX'
    STANDARD_TYPE = 'standard_type'

    # Dataype
    FILE_TYPE = 'file_type'
    FCS = 'FCS'
    CSV = 'CSV'
    INOCULATION_DENSITY = 'inoculation_density'
    INOCULATION_DENSITY_VALUE = INOCULATION_DENSITY + '.value'

    #IDs
    EXPERIMENT_ID = 'experiment_id'
    FILENAME = 'filename'
    SAMPLE_ID = 'sample_id'
    STRAIN_SBH_URI = STRAIN + '_sbh_uri'


    #Labs
    LAB = 'lab'
    TRANSCRIPTIC = 'Transcriptic'


    # Circuits
    STRAIN_CIRCUIT = STRAIN + '_circuit'
    STRAIN_INPUT_STATE = STRAIN + '_input_state'
    GATE = 'gate'
    INPUT = 'input'
    OUTPUT = 'output'
    XOR = 'XOR'
    XNOR = 'XNOR'
    OR= 'OR'
    NOR='NOR'
    NAND = 'NAND'
    AND = 'AND'

    #INPUTS #TODO: When you need to, use pydoe to construct these and create a dictionary of name variables
    INPUT_00 = '00'
    INPUT_01 = '01'
    INPUT_10 = '10'
    INPUT_11 = '11'
    
    REPLICATE = 'replicate'

