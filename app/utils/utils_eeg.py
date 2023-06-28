import mne

# TODO Maybe should change to handle all eeg file like fif and any other
import pandas as pd

from app.utils.utils_general import get_local_neurodesk_storage_path, get_single_file_from_neurodesk_interim_storage, \
    get_local_storage_path, get_single_file_from_local_temp_storage, load_data_from_csv


def load_data_from_edf(file_with_path):
    """This functions returns data from an edf file with the use of the MNE library
        This functions returns file with infer types enabled
    """
    data = mne.io.read_raw_edf(file_with_path, infer_types=True)
    return data


def load_data_from_edf_fif(file_with_path):
    """This functions returns data from an edf file with the use of the MNE library
        This functions returns file with infer types enabled
    """
    data = mne.io.read_raw_fif(file_with_path)
    return data

def load_data_from_edf_infer_off(file_with_path):
    """This functions returns data from an edf file with the use of the MNE library
        This functions returns file with infer types disabled
    """
    data = mne.io.read_raw_edf(file_with_path)
    return data

def load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id, ):
    if file_used == "printed":
        path_to_storage = get_local_neurodesk_storage_path(workflow_id, run_id, step_id)
        name_of_file = get_single_file_from_neurodesk_interim_storage(workflow_id, run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)
    else:
        # If not we use it from the directory input files are supposed to be
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    return data

def convert_yasa_sleep_stage_to_general(path_to_file):
    data = load_data_from_csv(path_to_file)
    data_to_add = data["Stage"]
    print(data_to_add)
    data_to_add = data_to_add.replace("W", "0")
    data_to_add = data_to_add.replace("N1", "1")
    data_to_add = data_to_add.replace("N2", "2")
    data_to_add = data_to_add.replace("N3", "3")
    data_to_add = data_to_add.replace("R", "4")
    print(data_to_add)
    new_csv = {"stage" : data_to_add.tolist() }
    print(new_csv)
    df = pd.DataFrame(new_csv)
    new_data = df.to_csv("C:\\neurodesktop-storage\\runtime_config\\workflow_1\\run_1\\step_5\\output\\new_old.csv", index=False)
    # print(new_data)



convert_yasa_sleep_stage_to_general("C:\\neurodesktop-storage\\runtime_config\\workflow_1\\run_1\\step_5\\output\\sleep_stage_confidence.csv")
