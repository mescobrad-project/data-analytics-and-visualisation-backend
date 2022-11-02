import mne

# TODO Maybe should change to handle all eeg file like fif and any other
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
