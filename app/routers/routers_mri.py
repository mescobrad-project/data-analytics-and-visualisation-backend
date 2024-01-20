import math
import os
import re
import time
import csv
import traceback
from fastapi import APIRouter, Query
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, lfilter, sosfilt, freqs, freqs_zpk, sosfreqz
from statsmodels.graphics.tsaplots import acf, pacf
from scipy import signal
import mne
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from fastapi.responses import JSONResponse
from os.path import isfile, join
import shutil
import tempfile
from app.utils.utils_general import create_local_step, get_all_files_from_local_temp_storage, \
    get_single_nii_file_from_local_temp_storage
from sqlalchemy.sql import text

from app.utils.utils_general import validate_and_convert_peaks, validate_and_convert_power_spectral_density, \
    create_notebook_mne_plot, get_neurodesk_display_id, get_local_storage_path, get_single_file_from_local_temp_storage, \
    NeurodesktopStorageLocation, get_local_neurodesk_storage_path

from app.utils.utils_mri import load_stats_measurements_table, load_stats_measurements_measures, plot_aseg

from app.utils.utils_datalake import upload_object

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import mne
from yasa import spindles_detect

import paramiko

from pydantic import BaseModel

from trino.auth import BasicAuthentication
from sqlalchemy import create_engine


router = APIRouter()

class FunctionOutputItem(BaseModel):
    """
    Known metadata information
    "files" : [["run_id: "string" , "step_id": "string"], "output":"string"]
     """
    workflow_id:str
    run_id: str
    step_id: str
    # file: str


@router.get("/list/mri/slices", tags=["list_mri_slices"])
async def list_mri_slices() -> dict:
    to_send_slices = []
    for it in range(1, 1000):
        to_send_slices.append("I" + str(it))
    return {'slices': to_send_slices}


@router.get("/list_nii_files")
async def list_nii_files(workflow_id: str, step_id: str, run_id: str):
    """ This functions list all nii files in the working directory"""
    # Get list of files from the local storage
    try:
        list_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
    except Exception as e:
        print(e)
        print("Error : Failed to retrieve file names")
        return []

    #     Remove not nii files
    for file in list_of_files:
        if not file.endswith(".nii"):
            list_of_files.remove(file)

    return {'files': list_of_files}


@router.get("/list_ita_files")
async def list_ita_files(workflow_id: str, step_id: str, run_id: str):
    """ This functions list all ita files in the working directory"""
    # Get list of files from the local storage
    try:
        list_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
    except Exception as e:
        print(e)
        print("Error : Failed to retrieve file names")
        return []

    #     Remove not ita files
    for file in list_of_files:
        if not file.endswith(".Ita"):
            print("Removing file: ", file)
            list_of_files.remove(file)

    return {'files': list_of_files}


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
async def return_free_surfer_recon(workflow_id: str,
                                   run_id: str,
                                   step_id: str,
                                   # input_test_name: str,
                                   file_name: str,
                                   ) -> dict:
    # Retrieve the paths file from the local storage
    # path_to_storage = NeurodesktopStorageLocation + '/runtime_config/workflow_' + str(workflow_id) + '/run_' + str(run_id) + '/step_' + str(step_id)
    path_to_storage = get_local_neurodesk_storage_path(workflow_id, run_id, step_id)
    path_to_file = get_local_storage_path(workflow_id, run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)

    # Connect to neurodesktop through ssh
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")

    channel = ssh.invoke_shell()
    response = channel.recv(9999)
    print(channel)
    print(channel.send_ready())

    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("rm ~/.license\n")
    channel.send("echo \"mkontoulis@epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\n")
    channel.send(" D4GXfOXX8hArD8mYfI4OhNCQ8Gb00sflXj1yH6NEFxk=\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output/test1\n")
    channel.send("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh\n")
    channel.send("export SUBJECTS_DIR=/home/user" + path_to_file + "/output\n")
    # channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output\n")
    #
    # channel.send("export SUBJECTS_DIR=/home/user" + path_to_storage + "/output\n")
    # channel.send("export SUBJECTS_DIR=" + path_to_storage + "\n")
    # channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output" + "\n")
    # channel.send("cd /neurodesktop-storage/freesurfer-output-2\n")
    # channel.send("cd " + path_to_file + "\n")
    # channel.send("ls > ls1.txt\n")

    # Get file name to open with EDFBrowser
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    # file_full_path = path_to_storage + "/" + name_of_file
    # channel.send("nohup freeview -v '/home/user" + file_full_path + "' &\n")
    #
    # channel.send("nohup freeview -v '/home/user" + path_to_file + "/" + name_of_file + "' &\n" )

    channel.send("cd /home/user" + path_to_file + "/\n")
    # channel.send(
    #     "sudo chmod a+rw /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")
    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "\n")

    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")

    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/output\n")

    channel.send("sudo mkdir -m777 ./output/samseg_output > mkdir.txt\n")

    channel.send(
        "nohup recon-all -subject " + file_name + " -i ./" + file_name + " -all > ./output/recon_log.txtr &\n")

    print("Name of file")
    print(file_name)
    print("all")
    # # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER
    # ssh = paramiko.SSHClient()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect("free-surfer", 22, username ="root" , password="freesurferpwd")
    #
    #
    # # Start recon COMMAND
    # ssh.exec_command("ls > ls.txt")
    # # ssh.exec_command("recon-all -subject subjectname -i /path/to/input_volume -T2 /path/to/T2_volume -T2pial -all")
    # # Redirect output to log.txt in output folder that has been created
    #
    # # If everything ok return Sucess
    to_return = "Success"
    return to_return

@router.get("/free_surfer/log/recon", tags=["return_free_surfer_recon"])
async def return_free_surfer_recon_log(workflow_id: str,
                                   run_id: str,
                                   step_id: str,
                                   ) -> dict:
    path_to_log = os.path.join( get_local_storage_path(workflow_id, run_id, step_id), "output", "recon_log.txtr")
    with open(path_to_log, "r") as f:
        lines = f.readlines()
        if lines[-1].rstrip() == "done":
            return True
        else:
            return False

    return False


@router.get("/free_surfer/log/samseg", tags=["return_free_surfer_samseg"])
async def return_free_surfer_samseg_log(workflow_id: str,
                                   run_id: str,
                                   step_id: str,
                                   ) -> dict:
    path_to_log = os.path.join( get_local_storage_path(workflow_id, run_id, step_id), "output", "samseg_log.txtr")
    with open(path_to_log, "r") as f:
        lines = f.readlines()

        last_line = lines[-1].rstrip()
        if re.search("^run_samseg complete.*", last_line):
            return True
        else:
            return False

    return False

@router.get("/free_surfer/log/vol2vol", tags=["return_free_surfer_samseg"])
async def return_free_surfer_log_vol2vol_coreg(workflow_id: str,
                                   run_id: str,
                                   step_id: str,
                                   output_file: str,
                                   ) -> dict:
    """Check if vol2vol or coreg has finished by checking if the output file exists"""
    path_to_output = os.path.join( get_local_storage_path(workflow_id, run_id, step_id), output_file)
    if os.path.exists(path_to_output):
        return True
    else:
        return False

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
async def return_free_surfer_samseg(workflow_id: str,
                                   run_id: str,
                                   step_id: str,
                                   input_file_name: str,
                                   input_flair_file_name: str | None = None,
                                    # input_slices: str,
                                    ) -> dict:
    # Retrieve the paths file from the local storage
    path_to_storage = get_local_neurodesk_storage_path(workflow_id, run_id, step_id)
    path_to_file = get_local_storage_path(workflow_id, run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)

    # Connect to neurodesktop through ssh
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")

    channel = ssh.invoke_shell()
    response = channel.recv(9999)
    print(channel)
    print(channel.send_ready())

    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("rm ~/.license\n")
    channel.send("echo \"mkontoulis@epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\n")
    channel.send(" D4GXfOXX8hArD8mYfI4OhNCQ8Gb00sflXj1yH6NEFxk=\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output/test1\n")
    channel.send("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh\n")
    channel.send("export SUBJECTS_DIR=/home/user" + path_to_file + "/output\n")
    # channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output\n")
    #
    # channel.send("export SUBJECTS_DIR=/home/user" + path_to_storage + "/output\n")
    # channel.send("export SUBJECTS_DIR=" + path_to_storage + "\n")
    # channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output" + "\n")
    # channel.send("cd /neurodesktop-storage/freesurfer-output-2\n")
    # channel.send("cd " + path_to_file + "\n")
    # channel.send("ls > ls1.txt\n")

    # Get file name to open with EDFBrowser
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    # file_full_path = path_to_storage + "/" + name_of_file
    # channel.send("nohup freeview -v '/home/user" + file_full_path + "' &\n")
    #
    # channel.send("nohup freeview -v '/home/user" + path_to_file + "/" + name_of_file + "' &\n" )

    # Move to working folder
    channel.send("cd /home/user" + path_to_file + "/\n")
    # channel.send("mkdir ./output/samseg_output\n")

    # Give permissions in working folder
    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "\n")

    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")

    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/output\n")

    # channel.send(
    #     "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/output/samseg_output\n")

    channel.send("sudo mkdir -m777 ./output/samseg_output > mkdir.txt\n")


    print("nohup run_samseg" + " --input " + input_file_name + " -o ./output/samseg_output > ./output/samseg_log.txtr &\n")

    if input_flair_file_name is None:
        channel.send(
            "nohup run_samseg" + " --input " + input_file_name + " -o ./output/samseg_output > ./output/samseg_log.txtr &\n")
    else:
        channel.send(
            "nohup run_samseg" + " --input " + input_file_name + " " + input_flair_file_name + " -o ./output/samseg_output > ./output/samseg_log.txtr &\n")

    # # If everything ok return Sucess
    to_return = "Success"
    return to_return

@router.get("/free_view/", tags=["return_free_view"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def return_free_view(input_test_name: str, input_slices: str,
                                   ) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")

    channel = ssh.invoke_shell()
    response = channel.recv(9999)
    print(channel)
    print(channel.send_ready())

    channel.send("cd /home/user/neurodesktop-storage\n")
    channel.send("nohup bash get_display.sh &\n")
    # channel.send(" ps aux| grep Xorg > dispplay.txt\n")
    # channel.send(" ps aux| grep Xorg > dispplay.txt\n")
    # channel.send("whoami > whoami.txt\n")
    # channel.send("declare -xp > env.txt\n")
    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # channel.send("nohup firefox &\n")
    channel.send("ls > ls.txt\n")
    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("echo \"mkontoulis @ epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
    channel.send("mkdir /neurodesktop-storage/freesurfer-output/test2\n")
    channel.send("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh\n")
    channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output\n")
    # channel.send("nohup freeview &\n")
    channel.send("mkdir neurodesktop-storage/screenshots\n")
    channel.send("cd neurodesktop-storage/screenshots\n")
    channel.send("nohup freeview -cmd ../commands.txt &\n")

    # If everything ok return Success
    to_return = "Success"
    return to_return


@router.get("/free_view/simple/", tags=["return_free_view"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def return_free_view_simple( workflow_id: str,
                                  step_id: str,
                                  run_id: str,
                                  file_to_open: str | None = None,
                                  file_to_open_2: str | None = None) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")

    channel = ssh.invoke_shell()
    response = channel.recv(9999)
    print(channel)
    print(channel.send_ready())
    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # channel.send("nohup firefox &\n")

    channel.send("pkill -INT freeview -u user\n")

    channel.send("ls > ls1.txt\n")
    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("echo \"mkontoulis @ epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
    channel.send("mkdir /neurodesktop-storage/freesurfer-output/test1\n")
    channel.send("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh\n")
    channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output\n")

    # Give permissions in working folder
    channel.send(
        "sudo chmod a+rw /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")

    # Get file name to open with EDFBrowser
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_nii_file_from_local_temp_storage(workflow_id, run_id, step_id)
    # file_full_path = path_to_storage + "/" + name_of_file

    channel.send("cd " + path_to_storage + "\n")

    # If no specific file name is provided open the first nii file in the folder
    if file_to_open is None:
        channel.send("nohup freeview -v" + name_of_file + " &\n")
    else:
        if(file_to_open_2 is not None):
            channel.send("nohup freeview -v " + file_to_open + " " + file_to_open_2 + " &\n")
        else:
            channel.send("nohup freeview -v " + file_to_open + " &\n")
    # channel.send("nohup freeview -v &\n")
    # channel.send("nohup freeview -v '/home/user" + file_full_path + "' &\n")

    # If everything ok return Success
    to_return = "Success"
    return to_return


@router.get("/free_surfer/coreg/", tags=["return_free_view"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def return_free_surfer_coreg( workflow_id: str,
                                  step_id: str,
                                  run_id: str,
                                  ref_file_name: str,
                                  flair_file_name: str,
                                  ) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")

    channel = ssh.invoke_shell()
    response = channel.recv(9999)
    print(channel)
    print(channel.send_ready())
    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # channel.send("nohup firefox &\n")


    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("rm ~/.license\n")
    channel.send("echo \"mkontoulis@epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\n")
    channel.send(" D4GXfOXX8hArD8mYfI4OhNCQ8Gb00sflXj1yH6NEFxk=\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output/test1\n")
    channel.send("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh\n")
    # channel.send("export SUBJECTS_DIR=/home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "\n")


    # # Give permissions in working folder
    # channel.send(
    #     "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")
    #
    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "\n")


    # Get file name to open with EDFBrowser
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    # reg_file_path = path_to_storage + "/" + reg_file_name
    # flair_file_path = path_to_storage + "/" + flair_file_name

    # Move to path
    # channel.send("ls > ls131231124124.txt\n")
    print("cd /home/user" + path_to_storage + "/\n")
    channel.send("cd /home/user" + path_to_storage + "/\n")
    # channel.send("sudo chmod 777 ./output/\n")
    channel.send("ls > ls1123.txt\n")
    # The created file has name "flairToT1_" + ref_file_name to keep track of it
    # print("nohup mri_coreg --mov "+ flair_file_name + " --ref " + ref_file_name + " --reg flairToT1_" + ref_file_name[:-4] + ".lta > logs_coreg.txt &\n")
    channel.send("nohup mri_coreg --mov " + flair_file_name + " --ref " + ref_file_name + " --reg flairToT1_" + ref_file_name[:-4] + ".lta > ./logs_coreg.txt &\n")

    # If everything ok return Success
    to_return = "Success"
    return to_return


@router.get("/free_surfer/vol2vol/", tags=["return_free_view"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def return_free_surfer_vol2vol( workflow_id: str,
                                  step_id: str,
                                  run_id: str,
                                  ref_file_name: str,
                                  flair_file_name: str,
                                  target_file_name: str,
                                  ) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")

    channel = ssh.invoke_shell()
    response = channel.recv(9999)
    print(channel)
    print(channel.send_ready())
    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # channel.send("nohup firefox &\n")

    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("rm ~/.license\n")
    channel.send("echo \"mkontoulis@epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\n")
    channel.send(" D4GXfOXX8hArD8mYfI4OhNCQ8Gb00sflXj1yH6NEFxk=\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
    channel.send("mkdir /neurodesktop-storage/freesurfer-output/test1\n")
    channel.send("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh\n")
    channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output\n")

    # Give permissions in working folder
    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")

    channel.send(
        "sudo chmod 777 /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "\n")

    # Get file name to open with EDFBrowser
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    flair_file_path = path_to_storage + "/" + flair_file_name
    target_file_path = path_to_storage + "/" + flair_file_name

    channel.send("cd /home/user" + path_to_storage + "/\n")

    # Output file has name "flair_reg_" + flair_file_name to keep track of it
    # Name of reg file is derived automatically from the name of the reference file as produced in the previous step
    # print("nohup mri_vol2vol --mov " + flair_file_name + " --reg flairToT1_" + ref_file_name[:-4] + ".lta --o flair_reg_" + ref_file_name + " --targ " + target_file_name + " &\n")
    channel.send("nohup mri_vol2vol --mov " + flair_file_name + " --reg flairToT1_" + ref_file_name[:-4] + ".lta --o flair_reg_" +  ref_file_name + " --targ " + target_file_name + " > ./log_vol2.txt &\n")

    # If everything ok return Success
    to_return = "Success"
    return to_return




@router.get("/free_surfer/", tags=["return_free_surfer"])
# Validation is done inline in the input of the function
# If file input is niifty the input_file parameter should contain the name and path of the file
# If on the other hand file input is DICOM the input_file parameter should contain the name of the Slice relevant
async def return_free_surfer(input_test_name: str, input_file: str,
                                   ) -> dict:
    # CONNECT THROUGH SSH TO DOCKER WITH FREESURFER
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username ="user" , password="password")

    channel = ssh.invoke_shell()
    response = channel.recv(9999)
    print(channel)
    print(channel.send_ready())

    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    channel.send("ls > ls2.txt\n")
    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("echo \"mkontoulis @ epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
    channel.send("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh\n")
    channel.send("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output\n")
    channel.send("cd /neurodesktop-storage/freesurfer-output\n")
    channel.send("nohup recon-all -subject " + input_test_name + " -i " + input_file + " -all > freesurfer_log.txtr &\n")

    # Start recon COMMAND
    # ssh.exec_command("ls > ls.txt")
    # ssh.exec_command("cd /neurocommand/local/bin/")
    # ssh.exec_command("./freesurfer-7_1_1.sh")
    # ssh.exec_command("echo \"mkontoulis @ epu.ntua.gr")
    # ssh.exec_command("60631")
    # ssh.exec_command(" *CctUNyzfwSSs")
    # ssh.exec_command(" FSNy4xe75KyK.\" >> ~/.license")
    # ssh.exec_command("export FS_LICENSE=~/.license")
    # ssh.exec_command("mkdir /neurodesktop-storage/freesurfer-output")
    # ssh.exec_command("source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh")
    # ssh.exec_command("export SUBJECTS_DIR=/neurodesktop-storage/freesurfer-output")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")

    # channel.send("cd /neurocommand/local/bin/" + ";"
    #                  + "./freesurfer-7_1_1.sh" + ";"
    #                  + "echo \"mkontoulis @ epu.ntua.gr" + ";"
    #                  + "60631" + ";"
    #                  + " *CctUNyzfwSSs" + ";"
    #                  + " FSNy4xe75KyK.\" >> ~/.license" + ";"
    #                  + "mkdir /neurodesktop-storage/freesurfer-output" + ";"
    #                  + "source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh" + ";"
    #                  + "freeview\n")


    # ssh.exec_command("cd /neurocommand/local/bin/" + ";"
    #                  + "./freesurfer-7_1_1.sh" + ";"
    #                  + "echo \"mkontoulis @ epu.ntua.gr" + ";"
    #                  + "60631" + ";"
    #                  + " *CctUNyzfwSSs" + ";"
    #                  + " FSNy4xe75KyK.\" >> ~/.license" + ";"
    #                  + "mkdir /neurodesktop-storage/freesurfer-output" + ";"
    #                  + "source /opt/freesurfer-7.1.1/SetUpFreeSurfer.sh" + ";"
    #                  + "freeview"
    #                  )
    # ssh.exec_command("recon-all -subject " + input_test_name + " -i " + input_slices + " -all")
    # ssh.exec_command("freeview")

    # ssh.exec_command("recon-all -subject subjectname -i /path/to/input_volume -T2 /path/to/T2_volume -T2pial -all")
    # Redirect output to log.txt in output folder that has been created

    # If everything ok return Success
    to_return = "Success"
    return to_return

@router.get("/return_samseg_result", tags=["return_samseg_stats"])
async def return_samseg_stats(workflow_id: str,
                            step_id: str,
                            run_id: str) -> []:
    path_to_file = get_local_storage_path(workflow_id, run_id, step_id)
    path_to_file = os.path.join(path_to_file, "output", "samseg_output", "samseg.stats")

    with open(path_to_file, newline="") as csvfile:
        if not os.path.isfile(path_to_file):
            return []
        reader = csv.reader(csvfile, delimiter=',')
        results_array = []
        i = 0
        for row in reader:
            i += 1
            temp_to_append = {
                "id": i,
                "measure": row[0].strip("# Measure "),
                "value": row[1],
                "unit": row[2]
            }
            results_array.append(temp_to_append)
        return results_array


@router.get("/return_reconall_stats/measures", tags=["return_all_stats"])
# Validation is done inline in the input of the function
async def return_reconall_stats_measures(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            file_name: str = None) -> dict:

    path_to_file = get_local_storage_path(workflow_id, run_id, step_id)
    path_to_file = os.path.join(path_to_file, "output", "ucl_test", "stats", file_name)
    stats_dict = load_stats_measurements_measures(path_to_file)
    return stats_dict

@router.get("/return_reconall_stats/table", tags=["return_all_stats"])
# Validation is done inline in the input of the function
async def return_reconall_stats_table(workflow_id: str,
                                             step_id: str,
                                             run_id: str,
                                             file_name: str = None) -> dict:

    path_to_folder = get_local_storage_path(workflow_id, run_id, step_id)
    if "*" in file_name:
        path_to_file = os.path.join(path_to_folder, "output", "ucl_test", "stats", "l" + file_name[1:])
        stats_dict = load_stats_measurements_table(path_to_file, 0, False)
        print(stats_dict["table"])
        path_to_file = os.path.join(path_to_folder, "output", "ucl_test", "stats", "r" + file_name[1:])
        right = load_stats_measurements_table(path_to_file, len(stats_dict["table"]), False)["table"]
        stats_dict["table"] = pd.concat([stats_dict["table"], right])
        print(stats_dict["table"])
    else:
        path_to_file = os.path.join(path_to_folder, "output", "ucl_test", "stats", file_name)
        stats_dict = load_stats_measurements_table(path_to_file, 0, False)

    stats_dict["table"] = stats_dict["table"].to_dict('records')
    return stats_dict

@router.get("/return_aseg_stats", tags=["return_aseg_stats"])
async def return_aseg_stats(workflow_id: str,
                                step_id: str,
                                run_id: str) -> str :
        path_to_file = get_local_storage_path(workflow_id, run_id, step_id)
        path_to_file = os.path.join(path_to_file, "output", "ucl_test", "stats", "aseg.stats")
        aseg = load_stats_measurements_table(path_to_file, 0)["table"]
        data = dict(zip(aseg['StructName'], pd.to_numeric(aseg['Volume_mm3'], errors='coerce')))
        plot_aseg(data, cmap='Spectral',
                    background='k', edgecolor='w', bordercolor='gray',
                    ylabel='Volume (mm3)', title='Volume of Subcortical Regions')

        return 'OK'

@router.put("/reconall_files_to_datalake")
async def reconall_files_to_datalake(workflow_id: str,
                                step_id: str,
                                run_id: str) -> str :
    try:
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        tmpdir = tempfile.mkdtemp()
        output_filename = os.path.join(tmpdir, 'ucl_test')
        print(output_filename)
        print(shutil.make_archive(output_filename, 'zip', root_dir=path_to_storage, base_dir='output/ucl_test'))
        upload_object(bucket_name="saved", object_name='expertsystem/workflow/'+ workflow_id+'/'+ run_id+'/'+
                                                          step_id+'/output/ucl_test.zip', file=output_filename + '.zip')

        return JSONResponse(content='zip file has been successfully uploaded to the DataLake', status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content='Error in saving zip file to the DataLake',status_code=501)

@router.put("/reconall_stats_to_trino")
#All recon-all stats to trino both tabular and measurements
async def reconall_stats_to_trino(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                patient_id: str) -> str:
    try:
        #patient id psakse an ginetai apo nifti
        #connect to trino
        TRINO__USR = "mescobrad-dwh-user"
        TRINO__PSW = "dwhouse"

        engine = create_engine(
            f"trino://{TRINO__USR}@trino.mescobrad.digital-enabler.eng.it:443/postgresql",
            connect_args={
                "auth": BasicAuthentication(TRINO__USR, TRINO__PSW),
                "http_scheme": "https",
            }
        )

        conn = engine.connect()

        # Check if patient is already on trino
        id_list = conn.execute("\
                       SELECT DISTINCT patient_id FROM postgresql.public.reconall_mri_tabular_stats")

        for id in id_list:
            if id[0] == patient_id:
                return JSONResponse(content="This patient's stats are already on trino", status_code=200)

        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        path_to_stats = os.path.join(path_to_storage, "output", "ucl_test", "stats")
        first = True

        #for all files with tabular data
        for filename in os.listdir(path_to_stats):
            path_to_file = os.path.join(path_to_stats, filename)
            if os.path.isfile(path_to_file):
                stats_dict = load_stats_measurements_table(path_to_file, 0)
                if not stats_dict["columns"]: continue

                #change to trino format
                stats_dict["table"] = stats_dict["table"].drop(columns={"Hemisphere", "id"}, errors="ignore")
                df = pd.melt(stats_dict["table"], id_vars=['StructName'], var_name='variable_name', value_name='variable_value')
                df["source"] = filename
                df["patient_id"] = patient_id
                df = df.rename(columns={"StructName": "rowid"})

                #concatenate to res
                if first:
                    res = df
                    first = False
                else:
                    res = pd.concat([res, df], ignore_index=True)

        #append to trino table
        res.to_sql(name='reconall_mri_tabular_stats', schema='public', con=conn, if_exists='append',
                 index=False, method='multi')

        #for all files with measurement data
        first = True
        for filename in os.listdir(path_to_stats):
            path_to_file = os.path.join(path_to_stats, filename)
            if os.path.isfile(path_to_file):
                stats_dict = load_stats_measurements_measures(path_to_file)
                if not stats_dict["measurements"]: continue

                stats_dict["dataframe"] = stats_dict["dataframe"].drop(columns={"Hemisphere"}, errors="ignore")

                #change to trino format
                stats_dict["dataframe"]["patient_id"] = patient_id
                df = pd.melt(stats_dict["dataframe"], id_vars=['patient_id'], var_name='variable_name', value_name='variable_value')
                df["source"] = filename
                #df = df.rename(columns={"StructName": "rowid"}) rowid or patient id ?

                #concatenate to res
                if first:
                    res = df
                    first = False
                else:
                    res = pd.concat([res, df], ignore_index=True)
        #append to trino table
        res.to_sql(name='reconall_mri_measurement_stats', schema='public', con=conn, if_exists='append',
                  index=False, method='multi')

        return JSONResponse(content='Stats have been successfully uploaded to Trino', status_code=200)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return JSONResponse(content='Error in uploading stats to Trino',status_code=501)

@router.get("/reconall_files_to_local")
async def reconall_files_to_local(workflow_id: str,
                                step_id: str,
                                run_id: str) -> str :
    try:
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        if not os.path.exists(path_to_storage + "/output/ucl_test"):
            create_local_step(workflow_id=workflow_id, step_id=step_id, run_id=run_id, files_to_download=[{'bucket' : 'saved', 'file' :
                'expertsystem/workflow/'+ workflow_id+'/'+ run_id+'/'+ step_id+'/output/ucl_test.zip', 'group_name': ''}])
            shutil.unpack_archive(path_to_storage + "/ucl_test.zip", path_to_storage)
            os.remove(path_to_storage + "/ucl_test.zip")
        return{'ok'}
    except Exception as e:
        print(e)
        return{'error'}
