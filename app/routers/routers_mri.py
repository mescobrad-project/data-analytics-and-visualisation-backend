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
@router.get("/free_surfer/recon", tags=["return_free_surfer_recon"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def return_free_surfer_recon(input_test_name: str, input_slices: str,
                                   ) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER

    # ENSURE COMMAND IS RUNNING

    # CREATE OUTPUT FOLDER SOMEHWERE BASED ON TEST NAME

    to_return = "Success"
    return to_return

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
