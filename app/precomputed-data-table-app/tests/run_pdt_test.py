"""
test running of main pdt script

:authors: Robert C. Moseley (robert.moseley@duke.edu)
"""

from precomputed_data_table.run_pdt import *
import pytest


class TestRunPDT(object):

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path_to_good_er_dirs = 'app/precomputed-data-table-app/tests/data/good_er_dirs'
        self.path_to_bad_er_dirs = 'app/precomputed-data-table-app/tests/data/bad_er_dirs'

    @pytest.fixture
    def exp_ref_exists(self):
        '''Returns an existing experiment reference in the tests/data/good_er_dirs'''
        return 'YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208'

    @pytest.fixture
    def exp_ref_does_not_exist(self):
        '''Returns an experiment reference not in the tests/data/good_er_dirs'''
        return 'YeastSTATES-CRISPR-Short-Duration-Time-Series-19870430'

    @pytest.fixture
    def good_exp_ref_dir(self):
        '''Returns an experiment reference directory with correct files'''
        dir_path = os.path.join(self.path_to_good_er_dirs,
                                'dc_YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208_20200328030858')
        return dir_path

    def test_latest_er_dir_exists(self, exp_ref_exists):
        expected_er_dir = 'dc_YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208_20200414095504'
        er_dir = get_latest_er(exp_ref_exists, self.path_to_good_er_dirs)
        assert er_dir == expected_er_dir

    def test_latest_er_dir_does_not_exist(self, exp_ref_does_not_exist):
        expected_msg = 'Experimental Reference not in Data Converge'
        er_dir = get_latest_er(exp_ref_does_not_exist, self.path_to_good_er_dirs)
        assert er_dir == expected_msg

    def test_record_json_exists(self, good_exp_ref_dir):
        expected_record_json_path = os.path.join(good_exp_ref_dir, 'record.json')
        record_json_path = return_er_record_path(good_exp_ref_dir)
        assert record_json_path == expected_record_json_path

    def test_record_json_does_not_exist(self):
        expected_msg = 'Experimental Reference status can not be determined. No record.json file found'
        bad_exp_ref = 'dc_YeastSTATES-CRISPR-Short-Duration-Time-Series-20191208_20200328030858'
        record_json_path = return_er_record_path(os.path.join(self.path_to_bad_er_dirs, bad_exp_ref))
        assert record_json_path == expected_msg

    def test_data_status_is_good(self, good_exp_ref_dir):
        record_json_path = os.path.join(good_exp_ref_dir, 'record.json')
        good_status = {"fc_raw": "ok", "fc_etl": "ok", "pr": "ok"}
        expected_msg = 'Data status 100% ok:\n{}'.format(good_status)
        status_msg = check_er_status(record_json_path)
        assert status_msg == expected_msg

    def test_data_status_is_not_good(self):
        bad_exp_ref = 'dc_YeastSTATES-OR-Gate-CRISPR-Dose-Response_20200423202039'
        record_json_path = os.path.join(self.path_to_bad_er_dirs, bad_exp_ref, 'record.json')
        bad_status = {"fc_raw": "Fail", "fc_etl": "ok. unique sample ids: 244",
                      "pr": "ok. unique sample ids: 310"}
        expected_msg = 'Data status NOT 100% ok:\n{}'.format(bad_status)
        status_msg = check_er_status(record_json_path)
        assert status_msg == expected_msg

    def test_fc_no_fc_etl_no_pr(self):
        good_exp_ref = 'dc_YeastSTATES-OR-Gate-CRISPR-Dose-Response_20200427004659'
        file_list = os.listdir(os.path.join(self.path_to_good_er_dirs, good_exp_ref))
        file_list = [f for f in file_list if 'fc_raw' in f]
        dtype_confirm_dict = confirm_data_types(file_list)
        expected_dtype_confirm_dict = {'platereader': False, 'fc_raw_log10': True,
                                       'fc_raw_events': True, 'fc_etl': False}
        assert dtype_confirm_dict == expected_dtype_confirm_dict

    def test_fc_etl_fc_no_pr(self):
        good_exp_ref = 'dc_YeastSTATES-OR-Gate-CRISPR-Dose-Response_20200427004659'
        file_list = os.listdir(os.path.join(self.path_to_good_er_dirs, good_exp_ref))
        file_list = [f for f in file_list if 'fc' in f]
        dtype_confirm_dict = confirm_data_types(file_list)
        expected_dtype_confirm_dict = {'platereader': False, 'fc_raw_log10': True,
                                       'fc_raw_events': True, 'fc_etl': True}
        assert dtype_confirm_dict == expected_dtype_confirm_dict

    def test_fc_etl_fc_pr(self):
        good_exp_ref = 'dc_YeastSTATES-OR-Gate-CRISPR-Dose-Response_20200427004659'
        file_list = os.listdir(os.path.join(self.path_to_good_er_dirs, good_exp_ref))
        dtype_confirm_dict = confirm_data_types(file_list)
        expected_dtype_confirm_dict = {'platereader': True, 'fc_raw_log10': True,
                                       'fc_raw_events': True, 'fc_etl': True}
        assert dtype_confirm_dict == expected_dtype_confirm_dict
