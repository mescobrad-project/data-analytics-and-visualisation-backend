def test_inactivity_masking_tests_FailMissingMaskingPeriodHour(test_app):
    response = test_app.get("/return_inactivity_mask_visualisation",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "inactivity_masking_period_hour": None,
                                    "inactivity_masking_period_hour": "20"})
    assert response.status_code == 422

def test_inactivity_masking_tests_Success(test_app):
    response = test_app.get("/return_inactivity_mask_visualisation",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "inactivity_masking_period_hour": "0",
                                    "inactivity_masking_period_minutes": "20"})
    assert response.status_code == 200

def test_add_mask_period_tests_FailMissingStartTime(test_app):
    response = test_app.get("/return_add_mask_period",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "mask_period_start": None,
                                    "mask_period_end": "2022-07-19 09:30:00"})
    assert response.status_code == 422

def test_add_mask_period_tests_Success(test_app):
    response = test_app.get("/return_add_mask_period",
                          params = {"workflow_id": "1",
                                    "run_id": "1",
                                    "step_id": "6",
                                    "mask_period_start": "2022-07-19 17:30:00",
                                    "mask_period_end": "2022-07-19 09:30:00"})
    assert response.status_code == 200
