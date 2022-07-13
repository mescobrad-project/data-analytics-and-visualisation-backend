import os
import shutil
import socket

import paramiko
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import routers_eeg, routers_mri, routers_datalake, routers_hypothesis

from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .utils.utils_general import create_neurodesk_user, read_all_neurodesk_users, \
    save_neurodesk_user, get_neurodesk_display_id, get_annotations_from_csv

tags_metadata = [
    {
        "name": "return_autocorrelation",
        "description": "return_autocorrelation function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_partial_autocorrelation",
        "description": "return_partial_autocorrelation function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_filters",
        "description": "return_filters function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "list_channels",
        "description": "test_list_channels function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_welch",
        "description": "return_welch function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_stft",
        "description": "return_stft function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_peaks",
        "description": "return_peaks function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_power_spectral_density",
        "description": "return_power_spectral_density function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_periodogram",
        "description": "return_periodogram function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_spindles_detection",
        "description": "return_spindles_detection function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
]

app = FastAPI(openapi_tags=tags_metadata)

# region CORS Setup
# This region enables FastAPI's built in CORSMiddleware allowing cross-origin requests allowing communication with
# the React front end
origins = [
    "http://localhost:3000",
    "http://localhost:3000/auto_correlation",
    "localhost:3000"
    "localhost:3000/auto_correlation"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="/neurodesktop-storage"), name="static")

# endregion


# region Routes of the application
@app.on_event("startup")
def initiate_functions():
    # Create folder in volume if it doesn't exist
    os.makedirs("/neurodesktop-storage/config", exist_ok=True)
    # Copy files from local storage to volume
    # Copy script for getting the current value of
    shutil.copy("neurodesk_startup_scripts/get_display.sh", "/neurodesktop-storage/config/get_display.sh")
    # Run the script with ssh from neurodesk
    # Initiate ssh connection with neurodesk container
    get_neurodesk_display_id()

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect("neurodesktop", 22, username="user", password="password")
        channel = ssh.invoke_shell()
        channel.send("cd /home/user/neurodesktop-storage\n")
        channel.send("sudo chmod 777 config\n")
        channel.send("cd /home/user/neurodesktop-storage/config\n")
        channel.send("sudo bash get_display.sh\n")
    except socket.gaierror:
        pass

@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello World"}


@app.get("/test/chart", tags=["root"])
async def root():
    return FileResponse('index.html')


@app.get("/test/read/users", tags=["root"])
async def test_read_users():
    # Test write user in local storage

    read_all_neurodesk_users()
    return "Success"

@app.get("/test/write/user", tags=["root"])
async def test_write_user(name, password):
    # Test write user in local storage

    save_neurodesk_user(name, password)
    return "Success"\

@app.get("/test/display/neurodesk", tags=["root"])
async def test_display_neurodesk():
    get_neurodesk_display_id()
    return "Success"

@app.get("/test/annotations/", tags=["root"])
async def test_annotations():
    get_annotations_from_csv()
    return "Success"

@app.get("/test/add/user", tags=["root"])
async def test_add_user(name, password):
    # Must add user both at ubuntu and at file of guacamole
    # 1 - Adding  at apache guacamole - needs sudo privileges
    # file etc/guacamole/user
    # 2 - Adding user at ubuntu
    # Done with ssh

    create_neurodesk_user(name, password)
    return "Success"

# Include routers from other folders
app.include_router(routers_eeg.router)
app.include_router(routers_mri.router)
app.include_router(routers_datalake.router)
app.include_router(routers_hypothesis.router)

# endregion
