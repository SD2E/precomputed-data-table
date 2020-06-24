import pandas as pd
from omics_tools import differential_expression, utils, comparison_generator
import json
import os

def qc_update(df, factors_to_keep, bool_factors, int_factors):
    df = df[ (df['QC_gcorr_BOOL']==True) & (df['QC_nmap_BOOL']==True) ]

    for col in df.columns:
        if col not in factors_to_keep and "QC_" in col:
            df.drop(col, axis=1, inplace=True)
    
    strain_column = ""
    for col in bool_factors:
        df[col] = df[col].astype(bool)
        
    for col in int_factors:
        df[col] = pd.to_numeric(df[col]).astype('int64')

    return df

def main(counts_df_path):
    counts_df = pd.read_csv(counts_df_path, sep=',', low_memory=False)
    counts_df.rename({counts_df.columns[0]:'sample_id'},inplace=True,axis=1)

    print(counts_df.shape)
    base_factor = ['Strain']
    sub_factors = ['Arabinose', 'Cuminic_acid', 'IPTG', 'Timepoint', 'Vanillic_acid', 'Xylose']
    factors_to_keep = base_factor + sub_factors
    print("factors_to_keep: {}".format(factors_to_keep))
    bool_factors=['IPTG','Arabinose']
    int_factors=['Timepoint']
    counts_df_qcd = qc_update(counts_df, factors_to_keep, bool_factors, int_factors)
    print(counts_df_qcd.shape)
    print(counts_df_qcd.head(5))
    counts_df_qcd.reset_index(inplace=True,drop=True)
    print("Unique strains: {}".format(len(counts_df_qcd['Strain'].unique())))

    DE_tests = ['Bacillus subtilis 168 Marburg', 'Bacillus subtilis 168 Marburg']
    run_dir = "./exp_ref_additive_design"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    groups_array = utils.group_by_factors(counts_df_qcd, factors_to_keep)
    comparison_indices = comparison_generator.generate_comparisons(counts_df_qcd,
                                                                   base_comparisons = DE_tests,
                                                                   base_factor = base_factor, 
                                                                   sub_factors = sub_factors,
                                                                   freedom = len(sub_factors),
                                                                   aggregation_flag = True,
                                                                   run_dir = run_dir,
                                                                   control_factor_in = {'Timepoint':5, 'IPTG':False, 'Arabinose':False})
    
    contrast_strings = differential_expression.make_contrast_strings(comparison_indices, groups_array)
    
if __name__ == '__main__':
    main("./experiment.ginkgo.29422_ReadCountMatrix_preCAD_transposed_filtered.csv")