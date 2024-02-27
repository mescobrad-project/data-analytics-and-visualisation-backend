def test_fml_tests_FailMissingDataset(test_app):
    response = test_app.get("/return_functional_linear_modelling",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "dataset": None})
    assert response.status_code == 422

def test_fml_tests_Success(test_app):
    response = test_app.get("/return_functional_linear_modelling",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "dataset": "0345-024_18_07_2022_13_00_00_New_Analysis"})
    assert response.status_code == 200

def test_fml_tests_FailWrongDataset(test_app):
    response = test_app.get("/return_functional_linear_modelling",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "dataset": "wrong_dataset"})

    assert response.status_code == 500
