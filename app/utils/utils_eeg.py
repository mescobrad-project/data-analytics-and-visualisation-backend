import mne

# TODO Maybe should change to handle all eeg file like fif and any other
import pandas as pd

from app.utils.utils_general import get_local_edfbrowser_storage_path, get_single_file_from_edfbrowser_interim_storage, \
    get_local_storage_path, get_single_file_from_local_temp_storage


def load_data_from_edf(file_with_path):
    """This functions returns data from an edf file with the use of the MNE library
        This functions returns file with infer types enabled
    """
    data = mne.io.read_raw_edf(file_with_path, infer_types=True)
    return data

def load_data_from_edf_infer_off(file_with_path):
    """This functions returns data from an edf file with the use of the MNE library
        This functions returns file with infer types disabled
    """
    data = mne.io.read_raw_edf(file_with_path)
    return data

def load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id, ):
    if file_used == "printed":
        path_to_storage = get_local_edfbrowser_storage_path(workflow_id, run_id, step_id)
        name_of_file = get_single_file_from_edfbrowser_interim_storage(workflow_id, run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)
    else:
        # If not we use it from the directory input files are supposed to be
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    return data
