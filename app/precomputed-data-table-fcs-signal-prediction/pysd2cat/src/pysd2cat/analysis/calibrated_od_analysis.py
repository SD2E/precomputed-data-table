import numpy as np

def get_per_experiment_statistics(df, pre='pre_OD600', post='OD600'):
    """
    Convert sample data in df into per experiment statistics.
    """
    experiment_groups = df.groupby(['experiment_id'])
    experiment_groups_result = \
        experiment_groups.agg({pre : [np.mean, np.std], 
                               post : [np.mean, np.std]}).reset_index()
    return experiment_groups_result

def get_per_experiment_statistics_by_od(df, pre='pre_OD600', post='OD600'):
    """
    Convert sample data in df into per experiment by od statistics.
    """
    experiment_od_groups = df.groupby(['experiment_id', 'od'])
    experiment_od_groups_result = \
        experiment_od_groups.agg({pre: [np.mean, np.std], 
                                  post : [np.mean, np.std]}).reset_index()
    return experiment_od_groups_result

def get_strain_statistics_by_od(df, pre='pre_OD600', post='OD600', group_cols=['strain', 'od', 'strain_circuit']):
    groups = df.groupby(group_cols)
    result = groups.agg({pre: [np.mean, np.std], 
                         post : [np.mean, np.std]}).reset_index()
    return result