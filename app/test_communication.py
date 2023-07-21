import os
from platform import python_version

import pytest
import validators
from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_initiate_function_eeg_1():
    response = client.put("/function/navigation/",
                           json={"workflow_id": "test_eeg_1",
                                    "run_id": "test_eeg_1",
                                    "step_id": "test_eeg_1",
                                 "function": "eeg_viewer",
                                 "metadata": {"files": [{'bucket': "saved", 'file': "ps_case_edf.edf"}]}
                                 })
    assert response.status_code == 200;

    response = response.json()

    assert "url" in response
    assert validators.url(response["url"])
    # assert response.json() == {"msg": "Hello World"}


def test_initiate_function_analysis_1():
    response = client.put("/function/navigation/",
                          json={"workflow_id": "test_analysis_1",
                                "run_id": "test_analysis_1",
                                "step_id": "test_analysis_1",
                                "function": "normality",
                                "metadata": {"files": [{'bucket': "demo", 'file': "expertsystem/workflow/3fa85f64-5717-4562-b3fc-2c963f66afa6/3fa85f64-5717-4562-b3fc-2c963f66afa6/3fa85f64-5717-4562-b3fc-2c963f66afa6/mescobrad_dataset.csv"}]}
                                })
    assert response.status_code == 200;

    response = response.json()

    assert "url" in response
    assert validators.url(response["url"])
    # assert response.json() == {"msg": "Hello World"}
