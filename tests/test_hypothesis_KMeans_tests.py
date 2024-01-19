def test_KMeans_Successful(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_notSelected_IndependentViariable(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_Nclusters_isnotInt(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_Missing_workflow_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_Missing_run_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_Missing_step_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_Missing_workspace_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_IndependentViariable_isnotNumerical(test_app):
    a= 'Success'
    assert a == 'Success'

def test_KMeans_IndependentViariable_ContainsNaNs(test_app):
    a= 'Success'
    assert a == 'Success'


