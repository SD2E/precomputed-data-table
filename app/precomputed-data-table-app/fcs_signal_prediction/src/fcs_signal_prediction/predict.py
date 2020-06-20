from pandas import DataFrame
from typing import Optional, List

from pysd2cat.analysis import correctness 

def predict_signal(df: DataFrame, 
                   experiment_identifier: str,
                   low_control : str,
                   high_control : str,
                   id_col : str,
                   channels : List[str],
                   strain_col : Optional[str]='strain_name') -> DataFrame:
          
    res = correctness.compute_predicted_output(df, 
                             training_df=None,
                             data_columns = channels, 
                             out_dir='.',
                             strain_col=strain_col,
                             high_control=high_control, 
                             low_control=low_control,
                             id_col=id_col,
                             use_harness=False,
                             description=None)
    
    return res
