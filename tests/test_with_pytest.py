import pytest

@pytest.fixture
def True_fixture():
    return True

@pytest.fixture
def False_fixture():
    return False

def test_always_passes(True_fixture):
    assert True_fixture==True

def test_always_fails(False_fixture):
    assert False_fixture==False
