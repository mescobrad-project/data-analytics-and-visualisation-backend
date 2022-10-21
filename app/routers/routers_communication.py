import json
import os

import requests
from fastapi import APIRouter, Request
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

router = APIRouter()

WFAddress = os.environ.get('WFAddress') if os.environ.get(
    'WFAddress') else "http://100.0.0.1:8000"

TestRunId = os.environ.get('TestRunId') if os.environ.get(
    'TestRunId') else "fe2b0997-6974-4fee-9178-a3291ea1744c"

TestStepId = os.environ.get('TestStepId') if os.environ.get(
    'TestStepId') else "db4a0c7d-0e09-49b2-ba80-7c5e4e3f0768"

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
    "samseg",
    "normality"
]


class FunctionNavigationItem(BaseModel):
    run_id: str
    step_id: str
    metadata: dict


# TODO
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

# TODO
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
    # url_to_redirect = "http:/"
    if navigation_item.metadata["function"]:
        match navigation_item.metadata["function"]:
            case "auto_correlation":
                # url_to_redirect += "/auto_correlation"
                url_to_redirect += "/eeg"
                url_to_redirect += "?eeg_function=auto_correlation"
            case "partial_auto_correlation":
                # url_to_redirect += "/partial_auto_correlation"
                url_to_redirect += "/eeg"
            case "filters":
                # url_to_redirect += "/filters"
                url_to_redirect += "/eeg"
            case "welch":
                # url_to_redirect += "/welch"
                url_to_redirect += "/eeg"
            case "find_peaks":
                # url_to_redirect += "/find_peaks"
                url_to_redirect += "/eeg"
            case "stft":
                # url_to_redirect += "/stft"
                url_to_redirect += "/eeg"
            case "periodogram":
                # url_to_redirect += "/periodogram"
                url_to_redirect += "/eeg"
            case "power_spectral_density":
                url_to_redirect += "/power_spectral_density"
            case "spindle":
                url_to_redirect += "/spindle"
            case "recon":
                url_to_redirect += "/freesurfer/recon"
            case "samseg":
                url_to_redirect += "/freesurfer/samseg"
            case "normality":
                url_to_redirect += "/normality_Tests/?file_path="+navigation_item.metadata["files"][0]+"&"
                os.makedirs('runtime_config/run_' + navigation_item.run_id + '_step_' + navigation_item.step_id, exist_ok=True)
                data_to_write = {
                }

                # TODO UNCOMMENT AND FIX
                # os.makedirs('runtime_config/run_' + navigation_item.run_id + '_step_' + navigation_item.step_id, exist_ok=True)
                # data_to_write = {
                # }
                #
                # if "files" in navigation_item.metadata:
                #     # TODO DOWNLOAD FILE instead of just saving file
                #     data_to_write["file"] = navigation_item.metadata["files"][0]
                #     with open('runtime_config/run_' + navigation_item.run_id + '_step_' + navigation_item.step_id + '/' + navigation_item.metadata["files"][0] + '.json', 'w', encoding='utf-8') as f:
                #         pass
                #
                # with open('runtime_config/run_' + navigation_item.run_id+ '_step_' + navigation_item.step_id + '/config_data.json', 'w', encoding='utf-8') as f:
                #     json.dump(data_to_write, f, ensure_ascii=False, indent=4)
    url_to_redirect += "?run_id="+ navigation_item.run_id+"&step_id=" + navigation_item.step_id
    # return RedirectResponse(url=url_to_redirect, status_code=301)
    print(url_to_redirect)
    return {"url": url_to_redirect}


@router.get("/function/existing", tags=["function_existing"], status_code=200)
async def task_existing(request: Request) -> dict:
    return {
        "analytics-functions": ExistingFunctions,
    }
