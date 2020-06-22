"""
test running of run_od_growth_analysis script

:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

from run_od_growth_analysis import *
import pytest

class TestRunODGrowthAnalysis(object):

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path_to_test_data_dir = 'tests/data'
        self.path_to_incomplete = 'tests/data/incomplete'
        self.path_to_complete = 'tests/data/complete'

    @pytest.fixture
    def exp_ref_exists(self):
        '''Returns an existing experiment reference in the tests/data/good_er_dirs'''
        return 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'

    @pytest.fixture
    def good_exp_ref_dir(self, exp_ref_exists):
        '''Returns an experiment reference directory with correct files'''
        dir_path = os.path.join(self.path_to_complete, exp_ref_exists, '20200610192131')
        return dir_path

    def test_pr_dataframe_col_names(self, exp_ref_exists, good_exp_ref_dir):

        pr_df = grab_pr_dataframe(exp_ref_exists, good_exp_ref_dir)
        pr_col_names = pr_df.columns.tolist()

        # for updated data converge test data
        expected_pr_cols = ['_id', 'sample_id', 'experiment_reference', 'experiment_id', 'lab',
                            'replicate', 'replicate_group', 'replicate_group_string',
                            'strain', 'strain_name', 'strain_class',
                            'media_type', 'inducer_type', 'inducer_concentration', 'inducer_concentration_unit',
                            'temperature', 'temperature_unit', 'timepoint', 'timepoint_unit',
                            'well', 'container_id', 'aliquot_id',
                            'od', 'fluor_gain_0.16', 'fluor_gain_0.16/od']

        assert sorted(pr_col_names) == sorted(expected_pr_cols)

    def test_meta_dataframe_col_names(self, exp_ref_exists, good_exp_ref_dir):

        meta_df = grab_meta_dataframe(exp_ref_exists, good_exp_ref_dir)
        meta_df_col_names = meta_df.columns.tolist()

        # for updated data converge test data
        expected_meta_cols = ['_id', 'sample_id', 'experiment_reference', 'experiment_id', 'lab',
                              'replicate', 'replicate_group', 'replicate_group_string',
                              'strain', 'strain_name', 'strain_class',
                              'media_type', 'inducer_type', 'inducer_concentration', 'inducer_concentration_unit',
                              'temperature', 'temperature_unit', 'timepoint', 'timepoint_unit',
                              'aliquot_id', 'total_counts', 'TX_plate_name', 'TX_project_name', 'TX_sample_name',
                              'flow_volume', 'well', 'flow_rate_uL/min', 'date_of_experiment', 'cells/mL']

        assert sorted(meta_df_col_names) == sorted(expected_meta_cols)

    def test_od_growth_analysis(self, exp_ref_exists, good_exp_ref_dir):

        pr_file_name = '__'.join([exp_ref_exists, 'platereader.csv'])
        pr_df = pd.read_csv(os.path.join(good_exp_ref_dir, pr_file_name))
        od_df = growth_analysis(pr_df, exp_ref_exists, '')
        expected_cols = ['experiment_id', 'well', 'strain', '_id', 'sample_id',
                         'experiment_reference', 'lab', 'replicate', 'replicate_group',
                         'replicate_group_string', 'strain_name', 'strain_class', 'media_type',
                         'inducer_type', 'inducer_concentration', 'inducer_concentration_unit',
                         'temperature', 'temperature_unit', 'timepoint', 'timepoint_unit',
                         'container_id', 'aliquot_id', 'od', 'fluor_gain_0.16',
                         'fluor_gain_0.16/od', 'dead', 'ungrowing', 'doubling_time', 'n0']

        assert od_df.columns.tolist() == expected_cols

    def test_rows_to_replicate_groups_od(self):
        filename = 'pdt_YeastSTATES-1-0-Growth-Curves__od_growth_analysis.csv'
        path_to_df = os.path.join(self.path_to_test_data_dir, filename)
        od_analysis_df = pd.read_csv(path_to_df, index_col=0)

        od_analysis_dropped_df = rows_to_replicate_groups(od_analysis_df, 'od')

        expected_cols = ['experiment_id', 'well', 'strain', 'lab', 'strain_name', 'media_type',
       'inducer_type', 'inducer_concentration', 'inducer_concentration_unit',
       'temperature', 'temperature_unit', 'od', 'dead', 'ungrowing',
       'doubling_time', 'n0', 'strain_class', 'experiment_reference']

        assert sorted(od_analysis_dropped_df.columns.tolist()) == sorted(expected_cols)

    def test_rows_to_replicate_groups_fc(self, good_exp_ref_dir):
        meta_fc_name = 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208__fc_meta.csv'
        path_to_df = os.path.join(good_exp_ref_dir, meta_fc_name)
        fc_meta_df = pd.read_csv(path_to_df)

        fc_meta_dropped_df = rows_to_replicate_groups(fc_meta_df, 'fc')

        expected_cols = ['TX_project_name', 'TX_sample_name', 'date_of_experiment', 'experiment_id',
                         'experiment_reference', 'flow_rate_uL/min', 'flow_volume',
                         'inducer_concentration', 'inducer_concentration_unit', 'inducer_type',
                         'lab', 'media_type', 'strain', 'strain_name', 'temperature',
                         'temperature_unit', 'total_counts', 'well', 'aliquot_id',
                         'cells/mL', 'strain_class']

        assert sorted(fc_meta_dropped_df.columns.tolist()) == sorted(expected_cols)
