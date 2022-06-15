from fastapi import APIRouter , Request
import requests
import os

router = APIRouter()

WFAddress = os.environ.get('WFAddress') if os.environ.get(
    'WFAddress') else "http://100.0.0.1:8000"

TestRunId = os.environ.get('TestRunId') if os.environ.get(
    'WFAddress') else "fe2b0997-6974-4fee-9178-a3291ea1744c"

TestStepId = os.environ.get('TestStepId') if os.environ.get(
    'WFAddress') else "db4a0c7d-0e09-49b2-ba80-7c5e4e3f0768"

ExistingFunctions = [
    "auto-correlation",
    "partial-auto-correlation",
    "filters",
    "welch",
    "find-peaks",
    "periodogram",
    "spindle",
    "recon-all",
    "samseg"
]
@router.get("/test/task/ping", tags=["test_task_ping"])
async def test_task_ping() -> dict:
    # channels = data.ch_names
    print(WFAddress)
    url = WFAddress + "/run/"+TestRunId + "/step/" + TestStepId + "/ping"
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


@router.post("/function/handling", tags=["function_handling"], status_code=200)
async def function_handling(request: Request) -> dict:
    # channels = data.ch_names
    print(request.json())
    return

@router.post("/function/existing", tags=["function_existing"], status_code=200)
async def task_existing(request: Request) -> dict:
    # channels = data.ch_names
    print(WFAddress)
    headers = {"Content-Type": "application/json"}
    data = {
        "functions": ExistingFunctions,
    }
    url = WFAddress + "/existing/functions" #Need to discuss this with EVOL
    print(url)
    response = requests.put(url=url, data=data, headers=headers)
    print("Test Response: Task Existing Sent INformation")
    print(response)

    print(request.json())
    return
