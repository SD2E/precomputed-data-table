from pysd2cat.analysis.threshold import compute_correctness
from pysd2cat.analysis.correctness import compute_correctness_harness
from pysd2cat.analysis.correctness import compute_correctness_all
import pandas as pd
import os

def test_threshold_correctness():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..',
                                   'resources',
                                   'r1c7cprv7fe49_r1c7jmje3ebhc.csv'))
    result_df = compute_correctness(df)
    assert 'probability_correct' in result_df.columns

def test_test_harness_correctness():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..',
                                   'resources',
                                   'r1c7cprv7fe49_r1c7jmje3ebhc.csv'))
    result_df = compute_correctness_harness(df, out_dir = os.getcwd())
    print(result_df)
    assert 'probability_correct' in result_df.columns

def test_multi_correctness_1():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..',
                                   'resources',
                                   'r1c7cprv7fe49_r1c7jmje3ebhc.csv'),
                    dtype={'output' : float, 'input' : object})
    result_df = compute_correctness_all(df, out_dir = os.getcwd())
    #print(result_df)
    assert 'mean_correct_threshold' in result_df.columns
    assert 'mean_correct_classifier' in result_df.columns

def test_multi_correctness_2():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..',
                                   'resources',
                                   'r1cgbw55mdzsa_r1cgp2v2jqn9h.csv'),
                    dtype={'od': float, 'input' : object, 'output' : float})
    result_df = compute_correctness_all(df, out_dir = os.getcwd())
    #print(result_df)
    assert 'mean_correct_threshold' in result_df.columns
    assert 'mean_correct_classifier' in result_df.columns
