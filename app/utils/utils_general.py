# DO NOT AUTO FORMAT THIS FILE THE STRINGS ADDED TO MNE NOTEBOOKS ARE TAB AND SPACE SENSITIVE
import json
import nbformat as nbf
import paramiko
import csv
import os

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


def create_notebook_mne_modular(file_to_save,
                                file_to_open,
                              notch_filter,
                                annotations,
                              bipolar_reference,
                              average_reference):
    """ Function that creates a mne jupyter notebook modularly
        Each input designates what should be added to the file
        The name of the file is currently given by the file_name parameter but that might change
    """
    nb = nbf.v4.new_notebook()
    nb['cells'] = []
    nb['cells'].append(nbf.v4.new_code_cell("""
import mne
import time

%matplotlib qt5

data = mne.io.read_raw_edf('""" + file_to_open + """"', infer_types=True, preload = True)
"""))

    if annotations:
        nb['cells'].append(nbf.v4.new_code_cell("""
while 1:
data.annotations.save(fname="annotation_test.csv", overwrite=True)
time.sleep(5)
        """))

    if bipolar_reference:
        nb['cells'].append(nbf.v4.new_code_cell("""
data = mne.set_bipolar_reference(data, anode=['"""+ bipolar_reference[0]+""""'], cathode=['"""+ bipolar_reference[1]+""""'])
        """))


    nbf.write(nb, "/neurodesktop-storage/mne/" + file_to_save + ".ipynb")


def create_notebook_mne_plot(run_id, step_id):
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
    nbf.write(nb, "/neurodesktop-storage/mne/" + run_id + "_" + step_id + '.ipynb')


def create_notebook_mne_plot_annotate(run_id, step_id):
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
    nbf.write(nb, "/neurodesktop-storage/" + run_id + "_" + step_id + '.ipynb')


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
        with open("/neurodesktop-storage/config/actual_display.txt", "r") as file:
            # Save lines in an array
            lines = file.read().splitlines()
            print(lines[0])
    except OSError as e:
        return "0"
    return lines[0]


def get_annotations_from_csv(annotation_file="annotation_test.csv"):
    """This function gets the annotation from the local storage and returns it as list of dicts"""
    with open("/neurodesktop-storage/"+ annotation_file, newline="") as csvfile:
        # Check if file exists
        if not os.path.isfile("/neurodesktop-storage/"+ annotation_file):
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
