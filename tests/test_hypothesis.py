# import pytest
from fastapi.testclient import TestClient
# from app.main import app
from app.main import app

# client = TestClient(app)

# @pytest.fixture
# def True_fixture():
#     return True
#
# @pytest.fixture
# def False_fixture():
#     return False
#
# def test_always_passes(True_fixture):
#     assert True_fixture==True
#
# def test_always_fails(False_fixture):
#     assert False_fixture==False

def test_return_columns(test_app):
    response = test_app.get("/return_columns",
                          params = {"file_name": '1000_test_cases_.csv',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"})
    assert response.status_code == 200

def test_return_columns_list(test_app):
    response = test_app.get("/return_columns",
                          params = {"file_name": '1000_test_cases_.csv',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"})
    response = response.json()
    assert type(response) == dict
    assert "columns" in response
    assert type(response["columns"]) == list

def test_return_columns_missing_workflow_id(test_app):
    response = test_app.get("/return_columns",
                          params = {"file_name": '1000_test_cases_.csv',
                                    # "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"})
    assert response.status_code == 422

def test_return_columns_file_None(test_app):
    response = test_app.get("/return_columns",
                          params = {"file_name": None,
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"})
    assert response.status_code == 200
    response = response.json()
    assert type(response) == dict
    assert "columns" in response
    assert type(response["columns"]) == list
