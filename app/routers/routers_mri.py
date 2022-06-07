import math

from fastapi import APIRouter, Query
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, lfilter, sosfilt, freqs, freqs_zpk, sosfreqz
from statsmodels.graphics.tsaplots import acf, pacf
from scipy import signal
import mne
import matplotlib.pyplot as plt
import mpld3
import numpy as np

from app.utils.utils_general import validate_and_convert_peaks, validate_and_convert_power_spectral_density

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import mne
from yasa import spindles_detect

router = APIRouter()


@router.get("/list/mri/slices", tags=["list_mri_slices"])
async def list_mri_slices() -> dict:
    to_send_slices = []
    for it in range(1, 1000):
        to_send_slices.append("I" + str(it))
    return {'slices': to_send_slices}


@router.get("/freesurfer/status/", tags=["list_freesurfer_status"])
# Placeholder function that returns results
# Will probably be replaced by return_free_surfer_recon_check function
async def list_freesurfer_status() -> dict:
    to_send_status = """ print greeting 
    "Hello stranger!"

print prompt
press "Enter" to continue
<user presses "Enter">

print call-to-action
    "How are you today?"

display possible responses 
    "1. Fine."
    "2. Great!"
    "3. Not good."

print request for input 
    "Enter the number that best describes you:"

if "1"
    print response
        "Dandy!"
if "2"
    print response
        "Fantastic!"
if "3"
    print response
        "Lighten up, buttercup!"

if input isn't recognized
    print response
        "You don't follow instructions very well, do you? """
    return {'status': to_send_status}


@router.get("/free_surfer/recon", tags=["return_free_surfer_recon"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def return_free_surfer_recon(input_test_name: str, input_slices: str,
                                   ) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER

    # Start recon COMMAND

    # Redirect output to log.txt in output folder that has been created

    # If everything ok return Sucess
    to_return = "Success"
    return to_return

@router.get("free_surfer/recon/check", tags=["return_free_surfer_recon"])
# Validation is done inline in the input of the function
# Check status of freesurfer function run
async def return_free_surfer_recon_check(input_test_name_check: str) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER

    # Enter OUTPUT FOLDER SOMEHWERE BASED ON TEST NAME

    # Get text of logs in this folder

    # If successfull return the contents of the folder otherwise errror

    to_return = "Logs"
    return {'status': to_return}


@router.get("/free_surfer/samseg", tags=["return_free_samseg"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def return_free_surfer_samseg(input_test_name: str, input_slices: str,
                                    ) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER

    # ENSURE COMMAND IS RUNNING

    # CREATE OUTPUT FOLDER SOMEHWERE BASED ON TEST NAME

    to_return = "Success"
    return to_return
