import os
import shutil
import socket

import paramiko
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import routers_eeg, routers_mri, routers_datalake, routers_hypothesis,  routers_communication, routers_actigraphy
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
    {
        "name": "return_samseg_stats",
        "description": "return results of Samseg function from Freesurfer",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_reconall_stats",
        "description": "return results of Recon-all function from Freesurfer",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "hypothesis_testing",
        "description": "return results of functions for Hypothesis testing",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "actigraphy_analysis",
        "description": "return results of functions for Actigraphy analysis",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_alpha_delta_ratio",
        "description": "return_alpha_delta_ratio function",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_asymmetry_indices",
        "description": "return_asymmetry_indices function",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
     },
     {
        "name": "return_alpha_variability",
        "description": "return_alpha_variability function",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_predictions",
        "description": "return_predictions function",
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
origins = ["*"]

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
    os.makedirs("/neurodesktop-storage", exist_ok=True)
    os.makedirs("/neurodesktop-storage/config", exist_ok=True)
    os.makedirs("/neurodesktop-storage/mne", exist_ok=True)

    # Create example files
    with open('annotation_test.csv', 'w') as fp:
        pass

    # Copy files from local storage to volume
    # Copy script for getting the current value of
    shutil.copy("neurodesk_startup_scripts/get_display.sh", "/neurodesktop-storage/config/get_display.sh")
    shutil.copy("neurodesk_startup_scripts/template_jupyter_notebooks/EDFTEST.ipynb", "/neurodesktop-storage/EDFTEST.ipynb")

    # CONERT WINDOWS ENDINGS TO UBUNTU / MIGHT NEED TO BE REMOVED AFTER VOLUME IS TRANSFERED TO NORMAL VOLUME AND NOT
    # BINDED
    # replacement strings
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'

    # relative or absolute file path, e.g.:
    file_path = r"/neurodesktop-storage/config/get_display.sh"

    with open(file_path, 'rb') as open_file:
        content = open_file.read()

    # Windows ➡ Unix
    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

    with open(file_path, 'wb') as open_file:
        open_file.write(content)

    # Run the script with ssh from neurodesk
    # Initiate ssh connection with neurodesk container
    # get_neurodesk_display_id()

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
    # Test read user in local storage

    read_all_neurodesk_users()
    return "Success"

@app.get("/test/write/user", tags=["root"])
async def test_write_user(name, password):
    # Test write user in local storage

    save_neurodesk_user(name, password)
    return "Success"

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
app.include_router(routers_communication.router)
app.include_router(routers_mri.router)
app.include_router(routers_hypothesis.router)
app.include_router(routers_datalake.router)
app.include_router(routers_actigraphy.router)

# endregion
