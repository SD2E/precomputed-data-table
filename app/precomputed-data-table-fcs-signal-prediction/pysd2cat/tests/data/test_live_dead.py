from pysd2cat.analysis.live_dead_analysis import add_live_dead, add_live_dead_test_harness
import pandas as pd
import os

def test_native_classifier():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..',
                                   'resources',
                                   'r1c7cprv7fe49_r1c7jmje3ebhc.csv'))
    result_df = add_live_dead(df)
    assert 'live' in result_df.columns

def test_test_harness_classifier():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..',
                                   'resources',
                                   'r1c7cprv7fe49_r1c7jmje3ebhc.csv'))
    result_df = add_live_dead_test_harness(df)
    assert 'live' in result_df.columns
