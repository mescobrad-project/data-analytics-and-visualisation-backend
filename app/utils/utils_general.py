import json
import nbformat as nbf
import paramiko


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


def create_notebook_mne_plot(run_id, step_id):
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


def create_newrodesk_user(user_name, user_password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")

    channel = ssh.invoke_shell()
    # Create a user in ubuntu
    channel.send("sudo useradd -p $(openssl passwd -1 " + user_password + ") " + user_name + "\n")
    # Add user in apache guacamole file
    channel.send("cd /etc/guacamole/\n")
    channel.send("sudo sed -i '$ d' user-mapping.xml")
    channel.send(""" 
    <authorize
            username=\"""" + user_name + """\"
            password=\"""" + user_password + """\"
            >
        <!-- Second authorized connection -->
        <connection name="Desktop Auto-Resolution (RDP)">
            <protocol>rdp</protocol>
            <param name="hostname">localhost</param>
            <param name="username">""" + user_name + """</param>
            <param name="password">""" + user_password + """</param>
            <param name="port">3389</param>
            <param name="security">any</param>
            <param name="ignore-cert">true</param>
            <param name="resize-method">reconnect</param>
            <param name="enable-drive">true</param>
            <param name="drive-oath">/home/user/Desktop</param>
        </connection>

    </authorize>
    """)
