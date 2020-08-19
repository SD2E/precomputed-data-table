import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
from harness.utils.names import Names
import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)

def main():
    # Reading in data from versioned-datasets repo.
    # Using the versioned-datasets repo is probably what most people want to do, but you can read in your data however you like.
    df = pd.read_csv('/Users/meslami/Documents/GitRepos/pysd2cat/src/data/tx_od-5.csv')

    
def predict(all_data_df,
            to_predict_data_df,
            output_location='.'):
    # list of feature columns to use and/or normalize:
    #Do these columns form a a unique entity? If not, we need to define a grouping.
    sparse_cols = ['growth_media_1', 'growth_media_2', \
                    'inc_temp', 'inc_time_1', 'inc_time_2',
                     'SynBioHub URI']
    #all_data_df['glycerol_stock'].fillna('blank',inplace=True)
    continuous_cols = ['post_od_raw']
    feature_cols = sparse_cols + continuous_cols
    l.debug(feature_cols)
    #cols_to_predict = []
    cols_to_predict = ['od']

    train1, test1 = train_test_split(all_data_df, test_size=0.2, random_state=5)

    th = TestHarness(output_location=output_location)

    th.run_custom(function_that_returns_TH_model=random_forest_regression,
                  dict_of_function_parameters={},
                  training_data=train1,
                  testing_data=test1,
                  data_and_split_description="OD Prediction",
                  cols_to_predict=cols_to_predict,
                  index_cols=['od', 'post_od_raw', 'SynBioHub URI'],
                  feature_cols_to_use=feature_cols,
                  normalize=True,
                  feature_cols_to_normalize=continuous_cols,
#                  feature_extraction=Names.RFPIMP_PERMUTATION,
                  feature_extraction=False,
                  predict_untested_data=to_predict_data_df,
                  sparse_cols_to_use=sparse_cols)


    l.debug("Extracting Test Harness Predictions ...")
    leader_board = pd.read_html(os.path.join(output_location, 'test_harness_results/custom_regression_leaderboard.html'))[0]
    leader_board = leader_board.sort_values(by=['Date', 'Time'], ascending=True)
    l.debug("Selecting run: " + str(leader_board.iloc[-1, :]))
    run = leader_board.loc[:,'Run ID'].iloc[-1]
    #print(run)
    run_path = os.path.join(output_location, 'test_harness_results/runs/', "run_" + run)
    predictions_path = os.path.join(run_path, 'predicted_data.csv')
    
    predictions_df = pd.read_csv(predictions_path, index_col=None, dtype={"index" : object})
    #predictions_df = predictions_df.set_index('class_label')
    return predictions_df


if __name__ == '__main__':
    main()
