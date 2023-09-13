from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)

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

def test_auto_correlation():
    response = client.get("/return_autocorrelation",
                          params={"file_used": "original",
                                  "workflow_id": "test_eeg_1",
                                    "run_id": "test_eeg_1",
                                    "step_id": "test_eeg_1",
                                  "input_name": "Fp1-AV"})
    assert response.status_code == 200
    response = response.json()

    # assert re
