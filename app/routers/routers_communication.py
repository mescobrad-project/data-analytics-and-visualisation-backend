import os

import requests
from fastapi import APIRouter, Request
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

router = APIRouter()

WFAddress = os.environ.get('WFAddress') if os.environ.get(
    'WFAddress') else "http://100.0.0.1:8000"

TestRunId = os.environ.get('TestRunId') if os.environ.get(
    'WFAddress') else "fe2b0997-6974-4fee-9178-a3291ea1744c"

TestStepId = os.environ.get('TestStepId') if os.environ.get(
    'WFAddress') else "db4a0c7d-0e09-49b2-ba80-7c5e4e3f0768"

ExistingFunctions = [
    "auto_correlation",
    "partial_auto_correlation",
    "filters",
    "welch",
    "find_peaks",
    "stft",
    "periodogram",
    "power_spectral_density",
    "spindle",
    "recon",
    "samseg"
]


class FunctionNavigationItem(BaseModel):
    run_id: str
    step_id: str
    metadata: list


@router.get("/test/task/ping", tags=["test_task_ping"])
async def test_task_ping() -> dict:
    # channels = data.ch_names
    print(WFAddress)
    url = WFAddress + "/run/" + TestRunId + "/step/" + TestStepId + "/ping"
    print(url)
    response = requests.get(url)
    print("Test Response: Task Ping")
    print(response)
    return {'test': "test"}


@router.get("/test/task/complete", tags=["test_task_complete"])
async def test_task_complete() -> dict:
    # channels = data.ch_names
    print(WFAddress)
    headers = {"Content-Type": "application/json"}
    data = {
        "action": "complete",
        "metadata": {}
    }
    url = WFAddress + "/run/" + TestRunId + "/step/" + TestStepId
    print(url)
    response = requests.put(url=url, data=data, headers=headers)
    print("Test Response: Task Ping")
    print(response)

    return {'test': "test"}


@router.put("/function/navigation/", tags=["function_navigation"])
async def function_navigation(navigation_item: FunctionNavigationItem) -> dict:
    # channels = data.ch_names
    # print("----RERERE----")
    # print(type(navigation_item))
    # print(navigation_item)

    url_to_redirect = "http://localhost:3000"
    if navigation_item.metadata[0]["function"]:
        match navigation_item.metadata[0]["function"]:
            case "auto_correlation":
                url_to_redirect += "/auto_correlation"
            case "partial_auto_correlation":
                url_to_redirect += "/partial_auto_correlation"
            case "filters":
                url_to_redirect += "/filters"
            case "welch":
                url_to_redirect += "/welch"
            case "find_peaks":
                url_to_redirect += "/find_peaks"
            case "stft":
                url_to_redirect += "/stft"
            case "periodogram":
                url_to_redirect += "/periodogram"
            case "power_spectral_density":
                url_to_redirect += "/power_spectral_density"
            case "spindle":
                url_to_redirect += "/spindle"
            case "recon":
                url_to_redirect += "/freesurfer/recon"
            case "samseg":
                url_to_redirect += "/freesurfer/samseg"
    url_to_redirect += "?run_id="+ navigation_item.run_id+"&step_id=" + navigation_item.step_id
    return RedirectResponse(url=url_to_redirect, status_code=303)


@router.get("/function/existing", tags=["function_existing"], status_code=200)
async def task_existing(request: Request) -> dict:
    return {
        "analytics-functions": ExistingFunctions,
    }
