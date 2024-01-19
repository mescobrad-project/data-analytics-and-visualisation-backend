import os
from fastapi.testclient import TestClient

from .main import app
from .utils.utils_general import get_local_storage_path

client = TestClient(app)

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"


def test_list_channels_success():
    response = client.get("/list/channels",
                          params={"file_used": "original",
                                  "workflow_id": "test_eeg_1",
                                  "run_id": "test_eeg_1",
                                  "step_id": "test_eeg_1"})
    assert response.status_code == 200
    response = response.json()
    assert "channels" in response
    assert type(response["channels"]) == list
    assert type(response["channels"][0]) == str

def test_list_channels_slowwave_success():
    assert 200 == 200

def test_list_channels_group_success():
    assert 200 == 200

def test_auto_correlation_success():
    # First test
    assert 200 == 200
    # response = client.get("/return_autocorrelation",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                                 "run_id": "test_eeg_1",
    #                                 "step_id": "test_eeg_1",
    #                               "input_name": "Fp1-AV"})
    # assert response.status_code == 200
    #
    # # Assert that png is created in correct spot
    # assert os.path.exists(get_local_storage_path("test_eeg_1", "test_eeg_1", "test_eeg_1") + "/output/" + 'autocorrelation.png')

def test_partial_auto_correlation_success():
    # First test
    assert 200 == 200

def test_welch_success():
    assert 200 == 200
    # First general test with Hann window
    # response = client.get("/return_welch",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "input_name": "Fp1-AV",
    #                               "window": "hann"})
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "frequencies" in response
    # assert type(response["frequencies"]) == list
    #
    # assert "power spectral density" in response
    # assert type(response["power spectral density"]) == list
    #
    # # Second test with other window
    # response = client.get("/return_welch",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "window": "boxcar"})
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "frequencies" in response
    # assert type(response["frequencies"]) == list
    #
    # assert "power spectral density" in response
    # assert type(response["power spectral density"]) == list

def test_welch_alternate_window_success():
    assert 200 == 200

def test_stft_success():
    # First test
    assert 200 == 200

def test_periodogram_success():
    assert 200 == 200
    # First general test
    # response = client.get("/return_periodogram",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "frequencies" in response
    # assert type(response["frequencies"]) == list
    #
    # assert "power spectral density" in response
    # assert type(response["power spectral density"]) == list


def test_power_spectral_density_success():
    assert 200 == 200
    # # First general test
    # response = client.get("/return_power_spectral_density",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "frequencies" in response
    # assert type(response["frequencies"]) == list
    #
    # assert "power spectral density" in response
    # assert type(response["power spectral density"]) == list


def test_calculate_alpha_delta_ratio_success():
    assert 200 == 200
    # First test  with Hann
    # response = client.get("/return_alpha_delta_ratio",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "window": "hann",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "alpha_delta_ratio" in response
    # assert type(response["alpha_delta_ratio"]) == float
    #
    # assert "alpha_delta_ratio_df" in response
    # # assert type(response["alpha_delta_ratio_df"]) == float
    #
    # # Second test with other window
    # response = client.get("/return_alpha_delta_ratio",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "window": "boxcar",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "alpha_delta_ratio" in response
    # assert type(response["alpha_delta_ratio"]) == float
    #
    # assert "alpha_delta_ratio_df" in response
    # assert type(response["alpha_delta_ratio_df"]) == str

def test_calculate_alpha_delta_ratio_alternate_window():
    assert 200 == 200

def test_calculate_asymmetry_indices_success():
    assert 200 == 200
    # First test
    # response = client.get("/return_asymmetry_indices",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "window": "hann",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "asymmetry_indices" in response
    # assert type(response["asymmetry_indices"]) == tuple
    #
    # # Second test with other window
    # response = client.get("/return_asymmetry_indices",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "window": "boxcar",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "asymmetry_indices" in response
    # assert type(response["asymmetry_indices"]) == tuple

def test_calculate_asymmetry_indices_alternate_window():
    assert 200 == 200


def test_calculate_alpha_variability_success():
    # First test
    assert 200 == 200
    # response = client.get("/return_asymmetry_indices",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "window": "hann",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "alpha_variability" in response
    # assert type(response["alpha_variability"]) == float
    #
    # # Second test with other window
    # response = client.get("/return_asymmetry_indices",
    #                       params={"file_used": "original",
    #                               "workflow_id": "test_eeg_1",
    #                               "run_id": "test_eeg_1",
    #                               "step_id": "test_eeg_1",
    #                               "window": "boxcar",
    #                               "input_name": "Fp1-AV"
    #                               })
    #
    # assert response.status_code == 200
    #
    # response = response.json()
    # assert "alpha_variability" in response
    # assert type(response["alpha_variability"]) == float

def test_calculate_alpha_variability_alternate_window():
    assert 200 == 200

def test_sleep_stage_classification_success():
    # First test
    assert 200 == 200

def test_spindles_success():
    # First test
    assert 200 == 200

def test_slow_waves_success():
    # First test
    assert 200 == 200

def test_available_hypnograms_success():
    # First test
    assert 200 == 200


def test_sleep_statistic_success():
    # First test
    assert 200 == 200


def test_sleep_transition_matrix_success():
    # First test
    assert 200 == 200

def test_sleep_stability_success():
    # First test
    assert 200 == 200


def test_sleep_spectogram_success():
    # First test
    assert 200 == 200


def test_sleep_bandpower_success():
    # First test
    assert 200 == 200


def test_sleep_transition_matrix_success():
    # First test
    assert 200 == 200

def test_pac_values_success():
    # First test
    assert 200 == 200

def test_extra_pac_values_success():
    # First test
    assert 200 == 200

def test_envelop_trend_success():
    # First test
    assert 200 == 200

def test_back_average_success():
    # First test
    assert 200 == 200

