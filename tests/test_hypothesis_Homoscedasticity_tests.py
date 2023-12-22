def test_Homoscedasticity_tests_Success(test_app):
    response = test_app.get("/check_homoscedasticity",
                          params = {"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--GAMES'],
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'center':'median',
                                    'name_of_test':"Levene"})
    assert response.status_code == 200
    response = response.json()
    assert "status" in response
    assert response["status"] == 'Success'

def test_Homoscedasticity_tests_column_None(test_app):
    response = test_app.get("/check_homoscedasticity",
                            params={"columns": None,
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'center':'median',
                                    'name_of_test': "Levene"})
    assert response.status_code == 200
    response = response.json()
    assert "status" in response
    assert response["status"] != 'Success'


def test_Homoscedasticity_tests_NameOfTest_NotInTheList(test_app):
    response = test_app.get("/check_homoscedasticity",
                            params={"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--GAMES'],
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'center':'median',
                                    'name_of_test': "NotInTheList"})
    assert response.status_code == 422

def test_Homoscedasticity_tests_Center_NotInTheList(test_app):
    response = test_app.get("/check_homoscedasticity",
                            params={"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--GAMES'],
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'center':'AA',
                                    'name_of_test': "Levene"})
    assert response.status_code == 422

def test_Homoscedasticity_tests_missing_workflow_id(test_app):
    response = test_app.get("/check_homoscedasticity",
                            params={"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--GAMES'],
                                    "workflow_id": None,
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'center':'median',
                                    'name_of_test': "Levene"})
    assert response.status_code == 422

def test_Homoscedasticity_tests_missing_run_id(test_app):
    response = test_app.get("/check_homoscedasticity",
                            params={"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--GAMES'],
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": None,
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'center':'median',
                                    'name_of_test': "Levene"})
    assert response.status_code == 422

def test_Homoscedasticity_tests_missing_step_id(test_app):
    response = test_app.get("/check_homoscedasticity",
                            params={"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--GAMES'],
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": None,
                                    'center':'median',
                                    'name_of_test': "Levene"})
    assert response.status_code == 422

def test_Homoscedasticity_tests_Missing_workspace_id(test_app):
    response = test_app.get("/normality_tests",
                            params={"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--GAMES'],
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": None,
                                    'center':'median',
                                    'name_of_test': "Levene"})
    assert response.status_code == 422

def test_Homoscedasticity_tests_SelectedVariable_isnotNumerical(test_app):
    response = test_app.get("/check_homoscedasticity",
                            params={"columns": ['1000_test_cases_.csv--CDRSUM','1000_test_cases_.csv--MOCALANX'],
                                    "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    "step_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                                    'center':'median',
                                    'name_of_test': "Levene"})
    assert response.status_code == 200
    response = response.json()
    assert "status" in response
    assert response["status"] != 'Success'
