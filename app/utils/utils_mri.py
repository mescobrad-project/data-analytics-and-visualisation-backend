import os
import pandas as pd

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

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

def load_stats_measurements(stats_path) -> dict:
    try:
        f = open(stats_path, "r")
    except Exception as e:
        print(e)
        print("File could not be opened")
        return pd.DataFrame()
    try:
        file_str = f.read()
        if "# ColHeaders" in file_str:
            file_list = file_str.split("# ColHeaders")
            list_of_lists = [row.split() for row in file_list[-1].split('\n')]
            df = pd.DataFrame(list_of_lists[1:-1], columns=list_of_lists[0])
        else:
            df = pd.DataFrame()

    except Exception as e:
        print(e)
        print("File has wrong format")
        return {"measurements": pd.DataFrame(), "table": df}
    try:
        measure_dict = {}
        lines = file_str.split('\n')
        for line in lines:
            if "# Measure" in line:
                measure = list(map(lambda elem : elem.strip(), line.split(",")))
                measure_dict[measure[-3]] = measure[-2] + ("" if measure[-1] == "unitless" else " " + measure[-1])
            elif "# hemi" in line:
                measure_dict["Hemisphere"] = line.split()[-1]
    except Exception as e:
        print("File has wrong format in Measure")
        print("Ignoring Measures...")
    return {"measurements": measure_dict, "table": df}
