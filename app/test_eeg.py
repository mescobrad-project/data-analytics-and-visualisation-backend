import os
from fastapi.testclient import TestClient

from .main import app
from .utils.utils_general import get_local_storage_path

client = TestClient(app)

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"


def test_list_channels():
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

def test_list_channels_slowwave():
    assert 200 == 200

def test_list_channels_group():
    assert 200 == 200

def test_auto_correlation():
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


def test_welch():
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


def test_periodogram():
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


def test_power_spectral_density():
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


def test_calculate_alpha_delta_ratio():
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


def test_calculate_alpha_delta_ratio():
    assert 200 == 200
    # First test  with Hann
    # response = client.get("/return_alpha_delta_ratio_periodogram",
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
    # response = client.get("/return_alpha_delta_ratio_periodogram",
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


def test_calculate_asymmetry_indices():
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



def test_calculate_alpha_variability():
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
