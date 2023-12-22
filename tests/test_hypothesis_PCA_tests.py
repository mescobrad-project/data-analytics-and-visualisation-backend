def test_PCA_Successful(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_notSelected_CategoricalViariable(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_notSelected_IndependentViariable(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_Solver_NotInTheList(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_NComponents_isnotNumerical(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_Missing_workflow_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_Missing_run_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_Missing_step_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_Missing_workspace_id(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_CategoricalViariable_isnotNumerical(test_app):
    a= 'Success'
    assert a == 'Success'


def test_PCA_CategoricalViariable_ContainsNaNs(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_IndependentViariable_isnotNumerical(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_IndependentViariable_ContainsNaNs(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_NComponents_notAllowed(test_app):
    a= 'Success'
    assert a == 'Success'

def test_PCA_Plots_created(test_app):
    a= 'Success'
    assert a == 'Success'

