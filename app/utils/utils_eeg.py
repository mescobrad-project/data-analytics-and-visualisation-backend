import os

import mne

# TODO Maybe should change to handle all eeg file like fif and any other
import pandas as pd

# from app.routers.routers_eeg import list_channels_slowwave
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
    """This function converts the sleep stage from the yasa automatic sleep staging to the general sleep staging format"""
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
    return  df
    # new_data = df.to_csv("C:\\neurodesktop-storage\\runtime_config\\workflow_1\\run_1\\step_5\\output\\new_old.csv", index=False)
    # print(new_data)


def transfer_hypnogram_to_interim_storage(workflow_id, run_id, step_id):
    """This function transfers the hypnogram from the local storage to the interim storage, called only when manual
    sleep stageing is called """
    path = get_local_storage_path(workflow_id, run_id, step_id)
    files_to_transfer = os.listdir(path)
    # Filtering only the files.
    files_to_transfer = [f for f in files_to_transfer if os.path.isfile(path + '/' + f)]

    for file in files_to_transfer:
        os.rename(path + "/" + file, get_local_neurodesk_storage_path(workflow_id, run_id, step_id) + "/" + file)


def return_number_and_names_groups(workflow_id, run_id, step_id):
    """This function returns information of groups available in a step
        These groups should always be found in the local storage of the step
    """
    path = get_local_storage_path(workflow_id, run_id, step_id)
    group_folders = os.listdir(path)
    # Filtering only the directories
    group_folders = [f for f in group_folders if os.path.isdir(path + '/' + f)]
    # Group folders all start with "group_"" so we keep only those
    group_folders = [f for f in group_folders if f.startswith("group_")]

    # Get the number of groups
    number_of_groups = len(group_folders)
    # Return
    return {"number_of_groups" : number_of_groups, "group_folders_name": group_folders}
    return number_of_groups, group_folders

def convert_generic_sleep_score_to_annotation(name_of_file, workflow_id, run_id, step_id):
    """This function converts the generic sleep score to annotations for the EDFBrowser"""
    # Load data from csv
    data_sleep_score = load_data_from_csv(get_local_storage_path(workflow_id, run_id, step_id) + "/" + name_of_file)

    # Convert dataframe to list
    data_sleep_score = data_sleep_score["stage"].values.tolist()

    #load edf file
    # edf_data = load_file_from_local_or_interim_edfbrowser_storage("original", workflow_id, run_id, step_id)

    # Initialise mne annotations
    annotations = mne.Annotations(onset=[], duration=[], description=[])

    current_time = 0
    print(data_sleep_score)
    for sleep_score in  data_sleep_score:
        if sleep_score == 0:
            annotations.append(onset=current_time, duration=30, description="W")
        elif sleep_score == 1:
            annotations.append(onset=current_time, duration=30, description="N1")
        elif sleep_score == 2:
            annotations.append(onset=current_time, duration=30, description="N2")
        elif sleep_score == 3:
            annotations.append(onset=current_time, duration=30, description="N3")
        elif sleep_score == 4:
            annotations.append(onset=current_time, duration=30, description="R")
        else:
            print("Error: Sleep score not recognised")

        current_time += 30
    print(annotations)

    # Save annoataions to file
    annotations.save(get_local_storage_path(workflow_id, run_id, step_id) + "/output/auto_hypno_annotations.txt", overwrite=True )
    annotations.save(get_local_storage_path(workflow_id, run_id, step_id) + "/neurodesk_interim_storage/auto_hypno_annotations.txt", overwrite=True )

    # edf_data.set_annotations(annotations)
    #
    # # Get the channel types
    # channel_types = edf_data.get_channel_types()
    #
    # # Get the physical maximum and minimum values for each channel type
    # lowest_minimum_value = 0.0
    # highest_maximum_value = 0.0
    # for ch_type in set(channel_types):
    #     if ch_type=='stim':
    #         continue
    #     ch_indices = [i for i, ch in enumerate(channel_types) if ch == ch_type]
    #     print(edf_data[ch_indices, :])
    #     for inner_ch in edf_data[ch_indices, :]:
    #         phys_min, phys_max = inner_ch.min(), inner_ch.max()
    #         print("Channel type:", ch_type)
    #         print("Physical minimum:", phys_min)
    #         print("Physical maximum:", phys_max)
    #         if phys_min < lowest_minimum_value:
    #             lowest_minimum_value = phys_min
    #         if phys_max > highest_maximum_value:
    #             highest_maximum_value = phys_max
    #
    #
    # print("Lowest minimum value:", lowest_minimum_value)
    # print("Highest maximum value:", highest_maximum_value)
    # # print(edf_data.get_data_range())
    #
    # mne.export.export_raw(raw= edf_data, fname=get_local_storage_path(workflow_id, run_id, step_id) + "/output/new_annot_edf.edf", physical_range=(lowest_minimum_value*1000000,highest_maximum_value*1000000), verbose=True, overwrite=True)


    # edf_data.save(path_to_storage = get_local_neurodesk_storage_path(workflow_id, run_id, step_id) + "/output/annotations.fif", overwrite=True)
    # print(new_data)
# mne.datasets.sample.data_path()
# convert_yasa_sleep_stage_to_general("C:\\neurodesktop-storage\\runtime_config\\workflow_1\\run_1\\step_5\\output\\sleep_stage_confidence.csv")
# convert_generic_sleep_score_to_annotation("C:\\neurodesktop-storage\\runtime_config\\workflow_1\\run_1\\step_5\\output\\new_old.csv", "1", "1", "1")

# list_channels_slowwave("1", "1", "1")
