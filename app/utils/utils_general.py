# DO NOT AUTO FORMAT THIS FILE THE STRINGS ADDED TO MNE NOTEBOOKS ARE TAB AND SPACE SENSITIVE
import json
import time
from os.path import isfile, join

import nbformat as nbf
import pandas as pd
import paramiko
import csv
import os
import mne
from mne.preprocessing import ICA
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler,DirCreatedEvent,FileCreatedEvent

from app.utils.utils_datalake import get_saved_dataset_for_Hypothesis

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"


def get_single_file_from_local_temp_storage(workflow_id, run_id, step_id):
    """Function to lazily retrieve name of file from local storage when there is a single file"""
    files_to_return = [f for f in os.listdir(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id) if isfile(join(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id, f))]
    return files_to_return[0]

def get_single_edf_file_from_local_temp_storage(workflow_id, run_id, step_id):
    """Function to lazily retrieve name of file from local storage when there is a single file"""
    files_to_return = [f for f in os.listdir(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id) if isfile(join(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id, f)) and (f.endswith(".edf"))]
    return files_to_return[0]

def get_single_file_from_neurodesk_interim_storage(workflow_id, run_id, step_id):
    """Function to lazily retrieve name and path of file from local storage when there is a single file"""
    files_to_return = [f for f in os.listdir(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id + '/neurodesk_interim_storage') if isfile(join(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id +'/neurodesk_interim_storage', f))]
    return files_to_return[0]


def get_files_for_slowwaves_spindle(workflow_id, run_id, step_id):
    """
        Function to retrieve files for the slowaves and spindles functions
        file name are returned as a dictionary with keys
        edf and csv respectively
        currently edf is detected by .edf or .fif
    """
    files_existing = [f for f in os.listdir(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id) if isfile(join(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id, f))]

    files_to_return = {}

    for file in files_existing:
        if file.endswith(".csv"):
            files_to_return["csv"] = file
        elif file.endswith(".edf") or file.endswith(".fif"):
            files_to_return["edf"] = file

    return files_to_return

def get_all_files_from_local_temp_storage(workflow_id, run_id, step_id):
    """Function to lazily retrieve name and path of files from local storage when there is a single file"""
    files_to_return = [f for f in os.listdir(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id) if isfile(join(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id, f))]
    return files_to_return

def get_output_info_path(workflow_id, run_id, step_id):
    """Function to retrieve path of info file of single run"""
    info_file_to_return = NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id + '/output/info.json'
    return info_file_to_return

def get_local_storage_path(workflow_id, run_id, step_id):
    """Function returns path with / at the end"""
    return NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id

def get_local_neurodesk_storage_path(workflow_id, run_id, step_id):
    """Function returns path with / at the end"""
    return NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id + '/neurodesk_interim_storage'

def load_data_from_csv(file_with_path):
    """This function read csv file using pandas library and return data as pandas dataframe
    """
    data = pd.read_csv(file_with_path)
    return data

def load_file_csv_direct(workflow_id, run_id, step_id ):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    data = load_data_from_csv(path_to_storage + "/" + name_of_file)
    return data


def create_local_step(workflow_id, run_id, step_id, files_to_download):
    """ files_to_download format is array of dicts with 'bucket' being bucket name 'file' object name each representing one file"""
    print("CREATING LOCAL STEP")
    print(files_to_download)
    path_to_save = NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(path_to_save + '/output', exist_ok=True)
    os.makedirs(path_to_save + '/neurodesk_interim_storage', exist_ok=True)
    # Download all files indicated
    for file_to_download in files_to_download:
        # print("file_to_download")
        # print(file_to_download[0])
        # print(file_to_download[1])
        # print(path_to_save + "/" +file_to_download[1])

        file_location_path = path_to_save + "/" +file_to_download["file"]
        if "/" in file_location_path:
            file_location_path = NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id + '/'+ file_location_path.split("/")[-1]

        print("file_location_path")
        get_saved_dataset_for_Hypothesis(bucket_name=file_to_download["bucket"], object_name=file_to_download["file"], file_location=file_location_path)
    # Info file might be unneeded
    with open( path_to_save + '/output/info.json', 'w', encoding='utf-8') as f:
        json.dump({"selected_datasets":files_to_download, "results":{}}, f)
        pass

def validate_and_convert_peaks(input_height, input_threshold, input_prominence, input_width, input_plateau_size):
    to_return = {
        "height": convert_string_to_number_or_array(input_height),
        "threshold": convert_string_to_number_or_array(input_threshold),
        "prominence": convert_string_to_number_or_array(input_prominence),
        "width": convert_string_to_number_or_array(input_width),
        "plateau_size": convert_string_to_number_or_array(input_plateau_size),
    }

    return to_return


def validate_and_convert_power_spectral_density(input_verbose):
    if input_verbose:
        if isinstance(input_verbose, bool):
            return input_verbose
        elif input_verbose.isdigit():
            return int(input_verbose)
        elif isinstance(input_verbose, str):
            return input_verbose
    else:
        return None


def convert_string_to_number_or_array(input):
    to_return = None
    if input:
        if input.isdigit():
            to_return = float(input)
        else:
            to_return = json.loads(input)
    return to_return


def create_notebook_mne_modular(
                                workflow_id,
                                run_id,
                                step_id,
                                file_to_save,
                                file_to_open,
                              notches_enabled,
                              notches_length,
                                annotations,
                              bipolar_references,
                              reference_type,
                                reference_channels_list,
                                selection_start_time,
                                selection_end_time,
                                repairing_artifacts_ica,
                                n_components,
                                list_exclude_ica,
                                ica_method):
    """ Function that creates a mne jupyter notebook modularly
        Each input designates what should be added to the file
        The name of the file is currently given by the file_name parameter but that might change
    """
    nb = nbf.v4.new_notebook()
    nb['cells'] = []
    nb['cells'].append(nbf.v4.new_code_cell("""
import mne
import time
import threading
from mne.preprocessing import ICA

%matplotlib qt5

data = mne.io.read_raw_edf('../""" + get_single_file_from_local_temp_storage(workflow_id, run_id, step_id) + """', infer_types=True, preload = True)
"""))

    if float(selection_start_time) != 0 or float(selection_end_time) != 0:
        nb['cells'].append(nbf.v4.new_code_cell("""
data.crop(float(""" + str(selection_start_time) +"""), float(""" + str(selection_end_time) + """))
"""))

    # If annotations are enabled
    if annotations:
        nb['cells'].append(nbf.v4.new_code_cell("""
def autosave_annots():
    data.annotations.save(fname="annotation_test.csv", overwrite=True)
    threading.Timer(5.0, autosave_annots).start()
"""))

    # IF bipolar references exist
    if bipolar_references:
        for bipolar_reference in bipolar_references:
            nb['cells'].append(nbf.v4.new_code_cell("""
data = mne.set_bipolar_reference(data, anode=['""" + bipolar_reference["anode"] + """'], cathode=['""" +
                                                bipolar_reference["cathode"] + """'])
"""))

    if notches_enabled:
        nb['cells'].append(nbf.v4.new_code_cell("""
data = data.notch_filter(freqs = """ + notches_length + """)
"""))

    if repairing_artifacts_ica:
        # Must check if n_components is int or float
        if n_components.isdigit():
            nb['cells'].append(nbf.v4.new_code_cell("""
data.load_data()
ica = ICA(n_components=int("""+n_components+"""), max_iter='auto', method=\""""+ica_method+"""\")
ica.fit(data)
ica
"""))
        else:
            nb['cells'].append(nbf.v4.new_code_cell("""
data.load_data()
ica = ICA(n_components=float(""" + n_components + """), max_iter='auto', method=\"""" + ica_method + """\")
ica.fit(data)
ica
"""))

        nb['cells'].append(nbf.v4.new_code_cell("""
data.load_data()
ica.plot_sources(data)
ica.exclude = [""" + ','.join(list_exclude_ica) + """]
reconst_raw = data.copy()
ica.apply(reconst_raw)

data.plot(n_channels=50)
"""))
        nb['cells'].append(nbf.v4.new_code_cell("""
reconst_raw.plot(n_channels=50)
"""))

    if reference_type == "average":
        nb['cells'].append(nbf.v4.new_code_cell("""
data = data.copy().set_eeg_reference(ref_channels='average')
"""))
    elif reference_type == "channels":
        nb['cells'].append(nbf.v4.new_code_cell("""
data.set_eeg_reference(ref_channels=[\"""" + '","'.join(reference_channels_list) + """\"])
"""))
    elif reference_type == "none":
        pass
    else :
        pass

    # We show the actual plot if not in artifact repair
    # Can send number of channels to be precise
    if not repairing_artifacts_ica:
        nb['cells'].append(nbf.v4.new_code_cell("""
fig = data.plot(n_channels=50)
"""))

        # Run the functions for annotations must always be in the end
        if annotations:
            nb['cells'].append(nbf.v4.new_code_cell("""
autosave_annots()
"""))

    print("PATH TO SAVE NOTEBOOK")
    print(get_local_neurodesk_storage_path(workflow_id, run_id, step_id) + "/" + file_to_save + ".ipynb")
    nbf.write(nb, get_local_neurodesk_storage_path(workflow_id, run_id, step_id) + "/" + file_to_save + ".ipynb")


def create_notebook_mne_plot(workflow_id, run_id, step_id):
    # Test Function to create sample mne notebook
    nb = nbf.v4.new_notebook()
    nb['cells'] = []
    nb['cells'].append(nbf.v4.new_code_cell("""import mne
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import pyplot
    import numpy as np
    %matplotlib qt5"""))
    nb['cells'].append(nbf.v4.new_code_cell("""
    data = mne.io.read_raw_edf('trial raw.edf', infer_types=True, preload = True)
    data = data.notch_filter(freqs = 70)
    fig = data.plot(n_channels=50)"""))
    nbf.write(nb, NeurodesktopStorageLocation + run_id + "_" + step_id + '.ipynb')


def create_notebook_mne_plot_annotate(workflow_id, run_id, step_id):
    nb = nbf.v4.new_notebook()
    nb['cells'] = []
    nb['cells'].append(nbf.v4.new_code_cell("""import mne
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import pyplot
    import numpy as np
    %matplotlib qt5"""))
    nb['cells'].append(nbf.v4.new_code_cell("""
    data = mne.io.read_raw_edf('trial raw.edf', infer_types=True, preload = True)
    data = data.notch_filter(freqs = 70)
    fig = data.plot(n_channels=50)"""))
    nbf.write(nb, NeurodesktopStorageLocation + run_id + "_" + step_id + '.ipynb')


def save_neurodesk_user(user_name, user_password):
    """This function save a new user in the local storage file, by appending a new line in the format
    'username:password \n'  """
    file = open("app_data/neurodesk_users.txt", "a")
    file.write(user_name + ":" + user_password+"\n")
    file.close()


def read_all_neurodesk_users():
    """ This function reads all users from the local storage and returns an array of array
    in the following format [[username,password]]"""
    with open("app_data/neurodesk_users.txt", "r") as file:
        # Save lines in an array
        lines = file.read().splitlines()
        for line_it, line in enumerate(lines):
            # Convert lines from string to array
            lines[line_it] = line.split(":")
        return lines


def create_neurodesk_user(user_name, user_password):
    """ This function creates a single new neurodesk user
    1) in the Ubuntu os running inside the docker
    2) in the apache guacamole server handling its interface allowing multiple concurrent users to access the vms
    3) in the local storage file in app_data/neurodesk_users.txt with format 'username:password \n'
    """
    # Initiate ssh connection with neurodesk container
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")
    channel = ssh.invoke_shell()

    # Create a user in ubuntu
    channel.send("sudo -s\n")
    channel.send("sudo adduser "+ user_name +" --gecos \"First Last,RoomNumber,WorkPhone,HomePhone\" --disabled-password" + "\n")
    channel.send("echo \"" + user_name + ":" + user_password + "\" | sudo chpasswd" + "\n")

    # Add user in apache guacamole file
    channel.send("sudo sed -i '$ d' /etc/guacamole/user-mapping.xml\n")

    channel.send("""sudo tee -a /etc/guacamole/user-mapping.xml <<EOF
<authorize username=\"""" + user_name + """\" password=\"""" + user_password + """\" >
<connection name="Command Line """ + user_name + """ (SSH)">
<protocol>ssh</protocol>
<param name="hostname">localhost</param>
<param name="username">""" + user_name + """</param>
<param name="password">""" + user_password + """</param>
<param name="port">22</param>
<param name="enable-sftp">true</param>
<param name="sftp-root-directory">/home/""" + user_name + """/Desktop</param>
</connection>
<connection name="Desktop Auto-Resolution """ + user_name + """ (RDP)">
<protocol>rdp</protocol>
<param name="hostname">localhost</param>
<param name="username">""" + user_name + """</param>
<param name="password">""" + user_password + """</param>
<param name="port">3389</param>
<param name="security">any</param>
<param name="ignore-cert">true</param>
<param name="resize-method">reconnect</param>
<param name="enable-drive">true</param>
<param name="drive-path">/home/""" + user_name + """/Desktop</param>
</connection>
</authorize>
</user-mapping> 
EOF\n""")

    # Add user in local folder
    save_neurodesk_user(user_name,user_password)
    return


def re_create_all_neurodesk_users():
    """ This function recreates all users in neurodesk in case the neurodesk container/pod is restarted since it
     doesnt store by itself any data"""
    saved_users = read_all_neurodesk_users()
    for saved_user in saved_users:
        create_neurodesk_user(saved_user[0], saved_user[1])

def get_neurodesk_display_id():
    """This function gets the id from the volume config folder where it was created when initiating the app"""
    try:
        with open(NeurodesktopStorageLocation + "/config/actual_display.txt", "r") as file:
            # Save lines in an array
            lines = file.read().splitlines()
            # print(lines)
            # print(NeurodesktopStorageLocation)
            # print(NeurodesktopStorageLocation + "/config/actual_display.txt")
    except OSError as e:
        return "0"

    if len(lines) > 0:
        # print(lines[0])
        return lines[0]
    else:
        return "0"


def get_annotations_from_csv(annotation_file="annotation_test.csv"):
    """This function gets the annotation from the local storage and returns it as list of dicts"""
    with open( NeurodesktopStorageLocation + "/" + annotation_file, newline="") as csvfile:
        # Check if file exists
        if not os.path.isfile(NeurodesktopStorageLocation + "/" + annotation_file):
            # if it doesnt return empty list
            return []

        # If it does read as csv and get contents
        reader = csv.reader(csvfile, delimiter=',')
        annotation_array = []
        first_flag = True
        for row in reader:
            if first_flag:
                # Stop first loop to not add headers
                first_flag = False
                continue
            temp_to_append = {
                "creator" : "user",
                "description" : row[2],
                "onset" : row[0],
                "duration" : row[1]
            }
            annotation_array.append(temp_to_append)
            # print(row)
            # print(', '.join(row))
        # print(annotation_array)
        return annotation_array
        # Save lines in an array
        # lines = file.read().splitlines()
        # print(lines[0])
    # return lines[0]


def write_function_data_to_config_file(parameter_data: dict | list, result_data: dict | list, workflow_id, run_id, step_id,):
    """This function gets the parameters and results of the applications' functions as either a list or dict with
    parameter_data
        <name_of_parameter>: <value_of_parameter>
        do we also need type of parameter?? like int or str
    result_data
        if result contains data:
            data_<name_of_result> -> <data>
        if results is file:
            path_<name_of_result> -> <path>
     and writes it to the config file in the output folder

     if parameter_data or result_data is a list then each entry represents a different iteration (only in functions where
     it's applicable to do multiple iterations with different data in the same run)
      and each entry in the list should contain one dict as described above
     """

    # Record misc information about the run
    run_data = {
       "workflow_id": workflow_id,
       "step_id": step_id,
       "run_id": run_id
    }

    if (type(parameter_data) is dict and type(result_data) is dict ) or (type(parameter_data) is list and type(result_data) is list):
        data_to_write = {"run_data": run_data,
                         "parameter_data": parameter_data,
                         "result_data": result_data}

        with open(get_output_info_path(workflow_id, run_id, step_id), "w") as info_file:
            info_file.seek(0)
            json.dump(data_to_write, info_file)
    else:
        return "error: parameter_data and result_data type should the same"





