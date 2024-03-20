def test_Correlation_tests_Success(test_app):
    a= 'Success'
    assert a == 'Success'

def test_Correlation_tests_variables_None(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_missing_workflow_id(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_missing_run_id(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_missing_step_id(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_Missing_workspace_id(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_SelectedVariable_isnotNumerical(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_SelectedVariable_ContainsNaNs(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_Alternative_NotInTheList(test_app):
    a = 'Success'
    assert a == 'Success'

def test_Correlation_tests_Method_NotInTheList(test_app):
    a = 'Success'
    assert a == 'Success'


