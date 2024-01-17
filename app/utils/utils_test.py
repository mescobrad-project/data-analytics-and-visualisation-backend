import os
from visbrain.gui import Sleep
from mne import io
from visbrain.io import download_file, path_to_visbrain_data

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

# Sleep().show()
# Sleep(data= NeurodesktopStorageLocation + '/PS case edf.edf').show()

# raw = io.read_raw_brainvision(vhdr_fname=dfile, preload=True)

#
# dfile = os.path.join(NeurodesktopStorageLocation, 'test', 'sub-02.vhdr')
# hfile = os.path.join(NeurodesktopStorageLocation, 'test', 'sub-02.hyp')
# cfile = os.path.join(NeurodesktopStorageLocation, 'test', 'excerpt2_config.txt')
#
# print(dfile)
# print(hfile)
# # Open the GUI :
# Sleep(data=dfile, hypno=hfile, config_file=cfile).show()
# Sleep().show()

#
# download_file("sleep_brainvision.zip", unzip=True, astype='example_data')
# target_path = path_to_visbrain_data(folder='example_data')
#
# dfile = os.path.join(target_path, 'sub-02.vhdr')
# hfile = os.path.join(target_path, 'sub-02.hyp')
# cfile = os.path.join(target_path, 'sub-02_config.txt')
#
# # Open the GUI :
# Sleep(data=dfile, hypno=hfile, config_file=cfile).show()




#
# download_file('sleep_edf.zip', unzip=True, astype='example_data')
# target_path = path_to_visbrain_data(folder='example_data')
#
# dfile = os.path.join(target_path, 'excerpt2.edf')
# hfile = os.path.join(target_path, 'Hypnogram_excerpt2.txt')
# cfile = os.path.join(target_path, 'excerpt2_config.txt')
#
# # Open the GUI :
# Sleep(data=dfile, hypno=hfile, config_file=cfile).show()








# download_file("sleep_brainvision.zip", unzip=True, astype='example_data')
# target_path = path_to_visbrain_data(folder='example_data')
#
# dfile = os.path.join(target_path, 'sub-02.vhdr')
# hfile = os.path.join(target_path, 'sub-02.hyp')
#
# # Read raw data using MNE-python :
# raw = io.read_raw_brainvision(vhdr_fname=dfile, preload=True)
#
# # Extract data, sampling frequency and channels names
# data, sf, chan = raw._data, raw.info['sfreq'], raw.info['ch_names']
#
# # Now, pass all the arguments to the Sleep module :
# Sleep(data=data, sf=sf, channels=chan, hypno=hfile).show()
