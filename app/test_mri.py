from fastapi.testclient import TestClient
import shutil
from .main import app

client = TestClient(app)

def test_return_samseg_stats():
    response = client.get("/return_samseg_result",
                          params={"workflow_id": "2",
                                  "run_id": "2",
                                  "step_id": "3"})
    assert response.status_code == 200
    response = response.json()
    assert type(response) == list
    for measure in response:
        assert all(key in measure for key in ("id", "measure", "value", "unit"))

def test_return_reconall_stats_measures():
    response = client.get("/return_reconall_stats/measures",
                          params={"workflow_id": "2",
                                  "run_id": "2",
                                  "step_id": "1",
                                  "file_name": "lh.aparc.a2009s.stats"})
    assert response.status_code == 200
    response = response.json()
    assert type(response) == dict
    assert type(response["measurements"]) == dict

    assert type(response["dataframe"]) == dict

    for measure in response["measurements"]:
        assert measure in response["dataframe"]

    for measure in response["dataframe"]:
        assert measure in response["measurements"]

def test_return_reconall_stats_table():
    response = client.get("/return_reconall_stats/table",
                          params={"workflow_id": "2",
                                  "run_id": "2",
                                  "step_id": "1",
                                  "file_name": "lh.aparc.a2009s.stats"})
    assert response.status_code == 200
    response = response.json()
    assert type(response) == dict

def test_return_aseg_stats():
    response = client.get("/return_aseg_stats",
                          params={"workflow_id": "2",
                                  "run_id": "2",
                                  "step_id": "1"})
    assert response.status_code == 200

def test_reconall_files_to_datalake():
    response = client.put("/reconall_files_to_datalake",
                          params={"workflow_id": "2",
                                  "run_id": "2",
                                  "step_id": "1"})
    assert response.status_code == 200
    assert response.content.decode('utf-8') == '"zip file has been successfully uploaded to the DataLake"'


def test_reconall_files_to_local():
    path = 'C:\\neurodesktop-storage\\runtime_config\\workflow_2\\run_2\\step_1'
    try:
        shutil.rmtree(path)
        print("Directory removed successfully")
    except OSError as o:
        print(f"Error, {o.strerror}: {path}")
    response = client.get("/reconall_files_to_local",
                          params={"workflow_id": "2",
                                  "run_id": "2",
                                  "step_id": "1"})
    assert response.status_code == 200
    assert response.json()[0] == 'ok'
