import datetime
import os
import time

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


def convert_compumedics_seconds_to_int(initial_seconds):
    """This function converts the seconds to an int format and rounds to upper second,
     it also returns the actual value it had as string to be added to the description"""
    split_initial = initial_seconds.split(":")
    seconds_to_return = 0
    actual_seconds_to_return = ""
    if len(split_initial) ==3:
        print("ERROR: DURATION EXCEEDS 60 Minutes")
        return
    elif len(split_initial) == 2:
        split_milliseconds = split_initial[1].split(".")
        if len(split_milliseconds) == 2:
            # Format is "m:ss.s"
            seconds_to_return = int(split_initial[0])*60 + int(split_milliseconds[0]) + 1
            actual_seconds_to_return = split_initial[1]
        elif len(split_milliseconds) == 1:
            # Format is "m:ss"
            # print(split_initial[0])
            # print(split_milliseconds)
            seconds_to_return = int(split_initial[0])*60 + int(split_milliseconds[0])
    to_return = {"seconds" : seconds_to_return, "actual_seconds_str": actual_seconds_to_return}

    return to_return
def convert_compumedics_to_annotation(file_hypno,file_eeg, workflow_id, run_id, step_id):
    data_df = pd.read_csv(get_local_storage_path(workflow_id, run_id, step_id) + "/" + file_hypno, sep=",", names=["start_time", "epoch", "stage", "description", "duration", "oxygen_sat", "ch_oxygen_sat", "position"])
    print(type(data_df))
    print(data_df)
    print(data_df["stage"])
    print(data_df["start_time"])
    data_times = data_df["start_time"].tolist()
    data_sleep_score = data_df["stage"].tolist()

    # Get initial time from eeg file and convert to seconds
    # TODO
    initial_time = data_times[0]
    initial_time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], initial_time.split(":")))

    # Day counter to know if day has lapsed into new one because given annotaitons only contain 24 Hour format
    day_counter = 0

    # Create Normal Annotations
    annotations = mne.Annotations(onset=[], duration=[], description=[])
    for index, entry in data_df.iterrows():
        # print(entry)
        converted_duration = convert_compumedics_seconds_to_int(entry["duration"])
        converted_onset = sum(x * int(t) for x, t in zip([3600, 60, 1], entry["start_time"].split(":")))
        converted_description = "Description:" + entry["description"] + " Oxygen Sat:" + entry["oxygen_sat"] + " Oxygen Sat Change:" + entry["ch_oxygen_sat"] + " Position: " + entry["position"] + " Actual Duration: " + entry["duration"]

        if converted_onset < initial_time_seconds:
            day_counter  += 1
        # print(converted_onset)
        # print(initial_time_seconds)
        print((converted_onset- initial_time_seconds) + (day_counter * 86400))
        print(converted_duration["seconds"])
        print(converted_description)
        # Onset is calculated by convertin time to seconds and subtracting
        annotations.append(onset=(converted_onset- initial_time_seconds) + (day_counter * 86400), duration= converted_duration["seconds"], description= converted_description)
    print("annotations")
    print(annotations)

    # Create Hypnogram
    # for index, entry in data_df.iterrows():


    return data_sleep_score


def convert_generic_sleep_score_to_annotation(name_of_file, workflow_id, run_id, step_id):
    """This function converts the generic sleep score to annotations for the EDFBrowser"""
    # Load data from csv
    # data_sleep_score = load_data_from_csv(get_local_storage_path(workflow_id, run_id, step_id) + "/" + name_of_file)
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
