import os
import pandas as pd
import paramiko

from app.utils.utils_general import get_neurodesk_display_id

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

NEURODESK_DEFAULT_USER = os.environ.get('NEURODESK_DEFAULT_USER') if os.environ.get('NEURODESK_DEFAULT_USER') else "random_user"
NEURODESK_DEFAULT_PASSWORD = os.environ.get('NEURODESK_DEFAULT_PASSWORD') if os.environ.get('NEURODESK_DEFAULT_PASSWORD') else "random_password"

def plot_aseg(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
              figsize=(15, 5), bordercolor='w', vminmax=[],
              title='', fontsize=15):
    """Plot subcortical ROI data based on the FreeSurfer `aseg` atlas

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the FreeSurfer `aseg` atlas. The full list
            of applicable regions can be found in the folder ggseg/data/aseg.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'aseg')
    reg = [op.basename(e) for e in glob(op.join(wd, '*'))]

    # Select data from known regions (prevents colorbar from going wild)
    known_values = []
    for k, v in data.items():
        if k in reg:
            known_values.append(v)

    whole_reg = ['Coronal', 'Sagittal']
    files = [open(op.join(wd, e)).read() for e in whole_reg]

    # A figure is created by the joint dimensions of the whole-brain outlines
    ax = ggseg._create_figure_(files, figsize, background,  title, fontsize, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg]
    ggseg._render_regions_(files, ax, bordercolor, edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = ggseg._get_cmap_(cmap, known_values, vminmax=vminmax)
    ggseg._render_data_(data, wd, cmap, norm, ax, edgecolor)

    # The following regions are ignored/displayed in gray
    NA = ['Cerebellum-Cortex', 'Cerebellum-White-Matter', 'Brain-Stem']
    files = [open(op.join(wd, e)).read() for e in NA]
    ggseg._render_regions_(files, ax, '#111111', edgecolor)

    # A colorbar is added
    ggseg._add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    fig = plt.gcf()
    fig.savefig(NeurodesktopStorageLocation + '/aseg.png')

def load_stats_measurements_table(stats_path, index_start, return_float=True) -> dict:
    try:
        f = open(stats_path, "r")
    except Exception as e:
        print(e)
        print("File could not be opened")
        return {"table": pd.DataFrame(), "columns": []}
    try:
        file_str = f.read()
        if "# ColHeaders" in file_str:
            file_list = file_str.split("# ColHeaders")
            list_of_lists = [row.split() for row in file_list[-1].split('\n')]
            if return_float:
                df = pd.DataFrame(list_of_lists[1:-1], columns=list_of_lists[0], dtype=float)
            else:
                df = pd.DataFrame(list_of_lists[1:-1], columns=list_of_lists[0])
            if "lh." in stats_path:
                df["Hemisphere"] = "Left"
            elif "rh." in stats_path:
                df["Hemisphere"] = "Right"

            columns = [
                {"field": name,
                 "headerName": name,
                 "flex": (2 if name=="StructName" else 1)} for name in df.columns]
            df.index += index_start
            df["id"] = df.index

        else:
            df = pd.DataFrame()
            columns = []
    except Exception as e:
        print(e)
        print("File has wrong format")
        return {"table": pd.DataFrame(), "columns": []}
    print(df)
    return {"table": df, "columns": columns}

def load_stats_measurements_measures(stats_path) -> dict:
    try:
        f = open(stats_path, "r")
        file_str = f.read()
    except Exception as e:
        print(e)
        print("File could not be opened")
        return {"measurements": {}, "dataframe": {}}
    try:
        measure_dict = {}
        measure_dataframe = {}
        lines = file_str.split('\n')
        for line in lines:
            if "# Measure" in line:
                measure = list(map(lambda elem : elem.strip(), line.split(",")))
                measure_dict[measure[-3]] = measure[-2] + ("" if measure[-1] == "unitless" else " " + measure[-1])
                measure_dataframe[measure[-3]] = [float(measure[-2])]
            elif "# hemi" in line:
                measure_dict["Hemisphere"] = line.split()[-1]
                measure_dataframe["Hemisphere"] = [line.split()[-1]]
    except Exception as e:
        print("File has wrong format in Measure")
        return {"measurements": {}, "dataframe": {}}
    measure_dataframe = pd.DataFrame.from_dict(measure_dataframe, dtype=float)
    return {"measurements": measure_dict, "dataframe": measure_dataframe}

"""
def download_mri_dataset(workflow_id, run_id, step_id, bucket, file):
    print("CREATING LOCAL STEP")
    print(file)
    path_to_save = NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(path_to_save + '/output', exist_ok=True)
    os.makedirs(path_to_save + '/neurodesk_interim_storage', exist_ok=True)
    # Download all files indicated
    file_location_path = path_to_save + "/" + file_to_download["file"]
        if "/" in file_location_path:
            file_location_path = NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id + '/'+ file_location_path.split("/")[-1]

        print("file_location_path")
        get_saved_dataset_for_Hypothesis(bucket_name=file_to_download["bucket"], object_name=file_to_download["file"], file_location=file_location_path)
    # Info file might be unneeded
    with open( path_to_save + '/output/info.json', 'w', encoding='utf-8') as f:
        json.dump({"selected_datasets":files_to_download, "results":{}}, f)
        pass
"""

def create_freesurfer_license():
    """ This functions creates a freesurfer license in the neurodesktop storage folder so there is access"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username=NEURODESK_DEFAULT_USER, password=NEURODESK_DEFAULT_PASSWORD)

    channel = ssh.invoke_shell()

    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # channel.send("cd /neurocommand/local/bin/\n")
    # channel.send("./freesurfer-7_3_2.sh\n")
    channel.send("cd /home/user\n")
    channel.send("sudo chmod a+rw /neurodesktop-storage\n")
    channel.send("cd /home/user/neurodesktop-storage\n")
    channel.send("rm .license\n")
    channel.send("echo \"mkontoulis@epu.ntua.gr\n")
    channel.send("60631\n")
    channel.send(" *CctUNyzfwSSs\n")
    channel.send(" FSNy4xe75KyK.\n")
    channel.send(" D4GXfOXX8hArD8mYfI4OhNCQ8Gb00sflXj1yH6NEFxk=\" >> .license\n")
    channel.send("export FS_LICENSE=/home/user/neurodesktop-storage/.license\n")
