import math
import os
import re
import time

from fastapi import APIRouter, Query
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, lfilter, sosfilt, freqs, freqs_zpk, sosfreqz
from statsmodels.graphics.tsaplots import acf, pacf
from scipy import signal
import mne
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from app.utils.utils_mri import plot_aseg

from app.utils.utils_general import validate_and_convert_peaks, validate_and_convert_power_spectral_density, \
    create_notebook_mne_plot, get_neurodesk_display_id, get_local_storage_path, get_single_file_from_local_temp_storage, \
    NeurodesktopStorageLocation, get_local_neurodesk_storage_path

from app.utils.utils_mri import load_stats_measurements

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import mne
from yasa import spindles_detect

import paramiko

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
async def return_free_surfer_recon(workflow_id: str,
                                   run_id: str,
                                   step_id: str,
                                   # input_test_name: str,
                                   # input_file: str,
                                   ) -> dict:
    # Retrieve the paths file from the local storage
    # path_to_storage = NeurodesktopStorageLocation + '/runtime_config/workflow_' + str(workflow_id) + '/run_' + str(run_id) + '/step_' + str(step_id)
    path_to_storage = get_local_neurodesk_storage_path(workflow_id, run_id, step_id)
    path_to_file = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)

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
    channel.send("cd /home/user/\n")
    channel.send("rm .license\n")

    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("echo \"mkontoulis@epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    channel.send("ls > ls1.txt\n")
    channel.send("sudo chmod -R a+rwx /home/user/neurodesktop-storage\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
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

    channel.send("cd " + path_to_file + "\n")
    # channel.send(
    #     "sudo chmod a+rw /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")
    channel.send(
        "nohup recon-all -subject " + name_of_file + " -i ./" + name_of_file + " -all > ./output/recon_log.txtr &\n")


    print("path_to_storage")
    print(path_to_storage)
    print("Name of file")
    print(name_of_file)
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
                                    # input_test_name: str,
                                    # input_slices: str,
                                    ) -> dict:
    # Retrieve the paths file from the local storage
    path_to_storage = get_local_neurodesk_storage_path(workflow_id, run_id, step_id)
    path_to_file = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)

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
    channel.send("cd /home/user/\n")
    channel.send("rm .license\n")

    channel.send("cd /neurocommand/local/bin/\n")
    channel.send("./freesurfer-7_1_1.sh\n")
    channel.send("echo \"mkontoulis@epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\" >> ~/.license\n")
    channel.send("export FS_LICENSE=~/.license\n")
    channel.send("ls > ls1.txt\n")
    channel.send("sudo chmod -R a+rwx /home/user/neurodesktop-storage\n")
    # channel.send("mkdir /neurodesktop-storage/freesurfer-output\n")
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

    channel.send("cd " + path_to_file + "\n")

    channel.send(
        "nohup run_samseg" + " --input " + name_of_file + " -o ./output/samseg_output > ./output/samseg_log.txtr &\n")

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
                                  file_to_open: str | None = None) -> dict:
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
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    file_full_path = path_to_storage + "/" + name_of_file

    # channel.send("nohup freeview -v &\n")
    channel.send("nohup freeview -v '/home/user" + file_full_path + "' &\n")

    # If everything ok return Success
    to_return = "Success"
    return to_return

# TODO freesuerfer/recon/
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

@router.get("/return_reconall_stats", tags=["return_all_stats"])
# Validation is done inline in the input of the function
async def return_aseg_stats(fs_dir: str = None, subject_id: str = None, file_name: str = None) -> dict:
    stats_dict = load_stats_measurements('example_data/stats/' + file_name)
    #data = dict(zip(aseg['StructName'], pandas.to_numeric(aseg['Volume_mm3'], errors='coerce')))
    return stats_dict

@router.get("/return_aseg_stats", tags=["return_aseg_stats"])
async def return_aseg_stats(fs_dir: str = None, subject_id: str = None) -> str:
    aseg = load_stats_measurements('example_data/aseg.stats')["table"]
    data = dict(zip(aseg['StructName'], pd.to_numeric(aseg['Volume_mm3'], errors='coerce')))
    plot_aseg(data, cmap='Spectral',
                    background='k', edgecolor='w', bordercolor='gray',
                    ylabel='Volume (mm3)', title='Volume of Subcortical Regions')

    return 'OK'
