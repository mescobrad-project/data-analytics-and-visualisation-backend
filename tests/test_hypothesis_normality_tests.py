def test_normality_tests_Success(test_app):
    response = test_app.get("/normality_tests",
                          params = {"column": '1000_test_cases_.csv--CDRSUM',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy':'propagate',
                                    'axis':0,
                                    'alternative':"two-sided",
                                    'name_test':"Shapiro-Wilk"})
    assert response.status_code == 200
    response = response.json()
    assert "status" in response
    assert response["status"] == 'Success'

def test_normality_tests_column_None(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": None,
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy': 'propagate',
                                    'axis': 0,
                                    'alternative': "two-sided",
                                    'name_test': "Shapiro-Wilk"})
    assert response.status_code == 422


def test_normality_tests_name_test_NotInTheList(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": '1000_test_cases_.csv--CDRSUM',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy': 'propagate',
                                    'axis': 0,
                                    'alternative': "two-sided",
                                    'name_test': "NotInTheList"})
    assert response.status_code == 422

def test_normality_tests_nan_policy_NotInTheList(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": '1000_test_cases_.csv--CDRSUM',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy': "NotInTheList",
                                    'axis': 0,
                                    'alternative': "two-sided",
                                    'name_test': "Shapiro-Wilk"})
    assert response.status_code == 422

def test_normality_tests_alternative_NotInTheList(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": '1000_test_cases_.csv--CDRSUM',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy': 'propagate',
                                    'axis': 0,
                                    'alternative': "NotInTheList",
                                    'name_test': "Shapiro-Wilk"})
    assert response.status_code == 422

def test_normality_tests_missing_workflow_id(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": '1000_test_cases_.csv--CDRSUM',
                                    "workflow_id": None,
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy': 'propagate',
                                    'axis': 0,
                                    'alternative': "two-sided",
                                    'name_test': "Shapiro-Wilk"})
    assert response.status_code == 422

def test_normality_tests_missing_run_id(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": '1000_test_cases_.csv--CDRSUM',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": None,
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy': 'propagate',
                                    'axis': 0,
                                    'alternative': "two-sided",
                                    'name_test': "Shapiro-Wilk"})
    assert response.status_code == 422

def test_normality_tests_missing_step_id(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": '1000_test_cases_.csv--CDRSUM',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": None,
                                    'nan_policy': 'propagate',
                                    'axis': 0,
                                    'alternative': "two-sided",
                                    'name_test': "Shapiro-Wilk"})
    assert response.status_code == 422

def test_normality_tests_error_in_results(test_app):
    response = test_app.get("/normality_tests",
                            params={"column": '1000_test_cases_.csv--MOCALANX',
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'nan_policy': 'propagate',
                                    'axis': 0,
                                    'alternative': "two-sided",
                                    'name_test': "Shapiro-Wilk"})
    assert response.status_code == 200
    response = response.json()
    assert "status" in response
    assert response["status"] != 'Success'
