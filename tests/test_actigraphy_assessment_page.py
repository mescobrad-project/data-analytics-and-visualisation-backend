def test_initial_dataset_tests_Success(test_app):
    response = test_app.get("/return_initial_dataset",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6"})
    assert response.status_code == 200

def test_initial_dataset_visualisation_tests_FailMissingWorkflowID(test_app):
    response = test_app.get("/return_daily_activity_activity_status_area",
                          params = {"workflow_id": None,
                                    "run_id": "1",
                                    "step_id": "6",
                                    "start_date": "2022/07/18 12:00:00",
                                    "end_date": "2022/07/20 12:00:00"})
    assert response.status_code == 422

def test_assessment_algorithm_tests_Success(test_app):
    response = test_app.get("/return_daily_activity",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "algorithm": "Crespo",
                                    "start_date": "2022/07/18 12:00:00",
                                    "end_date": "2022/07/20 12:00:00"})
    assert response.status_code == 200

def test_assessment_algorithm_tests_FailMissingAlgorithm(test_app):
    response = test_app.get("/return_daily_activity",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "algorithm": None,
                                    "start_date": "2022/07/18 12:00:00",
                                    "end_date": "2022/07/20 12:00:00"})
    assert response.status_code == 422

def test_final_dataset_visualisation_tests_FailMissingRunID(test_app):
    response = test_app.get("/return_final_daily_activity_activity_status_area",
                          params = {"workflow_id": "1",
                                    "run_id": None,
                                    "step_id": "6",
                                    "start_date": "2022/07/18 12:00:00",
                                    "end_date": "2022/07/20 12:00:00"})
    assert response.status_code == 422

def test_final_dataset_tests_Success(test_app):
    response = test_app.get("/return_final_dataset",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6"})
    assert response.status_code == 200
