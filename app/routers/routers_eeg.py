import csv
import os
import shutil
from datetime import datetime
import json
from functools import reduce
from os.path import isfile, join

from matplotlib import image as mpimg
from mne.stats import permutation_cluster_test
from tensorpac import Pac
import pingouin as pg
import seaborn as sns
import math
import yasa
from yasa import plot_spectrogram, spindles_detect, sw_detect, SleepStaging
import paramiko
from fastapi import APIRouter, Query
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, lfilter, sosfilt, freqs, freqs_zpk, sosfreqz
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf
from scipy import signal
from scipy.integrate import simps
from pmdarima.arima import auto_arima
import seaborn as seaborn
from yasa import SleepStaging
# import pywt
import mne
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
import lxml
from yasa import sleep_statistics
import logging

from app.utils.utils_eeg import load_data_from_edf, load_file_from_local_or_interim_edfbrowser_storage, \
    load_data_from_edf_fif, convert_yasa_sleep_stage_to_general, convert_generic_sleep_score_to_annotation
from app.utils.utils_general import validate_and_convert_peaks, validate_and_convert_power_spectral_density, \
    create_notebook_mne_plot, get_neurodesk_display_id, get_annotations_from_csv, create_notebook_mne_modular, \
    get_single_file_from_local_temp_storage, get_local_storage_path, get_local_neurodesk_storage_path, \
    get_single_file_from_neurodesk_interim_storage, write_function_data_to_config_file, \
    get_files_for_slowwaves_spindle, get_single_edf_file_from_local_temp_storage

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import mne
import requests
from yasa import spindles_detect
from pyedflib import highlevel
from app.pydantic_models import *

router = APIRouter( )

# region EEG Function pre-processing and functions
# TODO Finalise the use of file dynamically
data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)
NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

# data = mne.io.read_raw_fif("/neurodesktop-storage/trial_av_processed.fif")

#data = mne.io.read_raw_edf("example_data/psg1 anonym2.edf", infer_types=True)

# endregion

def rose_plot( workflow_id, run_id, step_id, angles, bins=12, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)

    plt.figure("rose_plot")
    ax = plt.subplot(projection='polar')

    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)

    # html_str = mpld3.fig_to_html(fig)
    # ax.savefig(NeurodesktopStorageLocation + '/rose_plot.png')
    # plt.show()
    # plt.savefig(NeurodesktopStorageLocation + '/rose_plot.png')
    plt.savefig(
        get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'rose_plot.png')
    return ax

def calcsmape(actual, forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast)))


def butter_lowpass(cutoff, fs, type_filter, order=5):
    if type_filter != 'bandpass':
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=type_filter, analog=False)
        return b, a
    else:
        nyq = 0.5 * fs
        low = cutoff[0] / nyq
        high = cutoff[1] / nyq
        b, a = butter(order, [low, high], btype=type_filter, analog=False)
        return b, a


def butter_lowpass_filter(data, cutoff, fs, type_filter, order=5):
    b, a = butter_lowpass(cutoff, fs, type_filter, order=order)
    y = lfilter(b, a, data)
    return y


@router.get("/list/channels", tags=["list_channels"])
async def list_channels(workflow_id: str,
                        step_id: str,
                        run_id: str,
                        file_used: str | None = Query("original",
                                        regex="^(original)$|^(printed)$"),
                        ) -> dict:

    # If file is altered we retrieve it from the edf interim storage fodler
    if file_used == "printed":
        path_to_storage = get_local_neurodesk_storage_path(workflow_id, run_id, step_id)
        name_of_file = get_single_file_from_neurodesk_interim_storage(workflow_id, run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)
    else:
        # If not we use it from the directory input files are supposed to be
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    channels = data.ch_names
    return {'channels': channels}


# TODO Functions might need change in future check it afterwards
@router.get("/list/channels/slowwave", tags=["list_channels"])
async def list_channels_slowwave(
                        workflow_id: str,
                        step_id: str,
                        run_id: str
                        ) -> dict:

    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    data = load_data_from_edf_fif(path_to_storage + "/" + files["edf"])

    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    # data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    channels = data.ch_names
    return {'channels': channels}



@router.get("/return_autocorrelation", tags=["return_autocorrelation"])
# Validation is done inline in the input of the function
async def return_autocorrelation(workflow_id: str, step_id: str, run_id: str,
                                 input_name: str,
                                 input_adjusted: bool | None = False,
                                 input_qstat: bool | None = False,
                                 input_fft: bool | None = False,
                                 input_bartlett_confint: bool | None = False,
                                 input_missing: str | None = Query("none",
                                                                   regex="^(none)$|^(raise)$|^(conservative)$|^(drop)$"),
                                 input_alpha: float | None = None,
                                 input_nlags: int | None = None,
                                 file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                 ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            z = acf(raw_data[i], adjusted=input_adjusted, qstat=input_qstat,
                    fft=input_fft,
                    bartlett_confint=input_bartlett_confint,
                    missing=input_missing, alpha=input_alpha,
                    nlags=input_nlags)

            to_return = {
                'values_autocorrelation': None,
                'confint': None,
                'qstat': None,
                'pvalues': None
            }

            fig, ax = plt.subplots(nrows=1, ncols=1, facecolor="#F0F0F0")

            ax.legend(["ACF"], loc="upper right", fontsize="x-small", framealpha=1, edgecolor="black", shadow=None)
            ax.grid(which="major", color="grey", linestyle="--", linewidth=0.5)
            print(z[0])

            # Parsing the results of acf into a single object
            # Results will change depending on our input
            if input_qstat and input_alpha:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['confint'] = z[1].tolist()
                to_return['qstat'] = z[2].tolist()
                to_return['pvalues'] = z[3].tolist()
                # plot_acf(z, adjusted=input_adjusted, alpha=input_alpha, lags=len(z[0].tolist())-1, ax=ax)
                # ax.set_xticks(np.arange(1, len(z[0].tolist()), step=1))
            elif input_qstat:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['qstat'] = z[1].tolist()
                to_return['pvalues'] = z[2].tolist()
                # plot_acf(z, adjusted=input_adjusted, lags=len(z[0].tolist())-1, ax=ax)
                # ax.set_xticks(np.arange(1, len(z[0].tolist()), step=1))
            elif input_alpha:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['confint'] = z[1].tolist()
                plot_acf(x=raw_data[i],
                         adjusted=input_adjusted,
                         # qstat=input_qstat,
                         fft=input_fft,
                         bartlett_confint=input_bartlett_confint,
                         missing=input_missing,
                         alpha=input_alpha,
                         lags=input_nlags,
                         ax=ax,
                         use_vlines=True)
                # plot_acf(z, adjusted=input_adjusted, alpha=input_alpha, lags=len(z[0].tolist()) -1, ax=ax)
                # ax.set_xticks(np.arange(1, len(z[0].tolist()), step=1))
            else:
                to_return['values_autocorrelation'] = z.tolist()
                # plot_acf(z, adjusted=input_adjusted, lags=len(z.tolist())-1, ax=ax)
                # plot_acf(x=raw_data[i], adjusted=input_adjusted, qstat=input_qstat,
                #     fft=input_fft,
                #     bartlett_confint=input_bartlett_confint,
                #     missing=input_missing, alpha=input_alpha,
                #     nlags=input_nlags, ax=ax, use_vlines=True)
                plot_acf(x=raw_data[i],
                        adjusted=input_adjusted,
                        # qstat=input_qstat,
                        fft=input_fft,
                        bartlett_confint=input_bartlett_confint,
                        missing=input_missing, alpha=input_alpha,
                        ax=ax,
                        lags=input_nlags,
                        use_vlines=True)
                # ax.set_xticks(np.arange(1, len(z.tolist()), step=1))

            # plt.show()
            print("RETURNING VALUES")
            print(to_return)
            plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'autocorrelation.png')

            # plt.show()

            # Prepare the data to be written to the config file
            parameter_data = {
                'name': input_name,
                'adjusted': input_adjusted,
                'qstat': input_qstat,
                'fft': input_fft,
                'bartlett_confint': input_bartlett_confint,
                'missing': input_missing,
                'alpha': input_alpha,
                'nlags': input_nlags,
            }
            result_data = {
                'data_values_autocorrelation': to_return['values_autocorrelation'],
                'data_confint': to_return['confint'],
                'data_qstat': to_return['qstat'],
                'data_pvalues': to_return['pvalues']
            }

            write_function_data_to_config_file(parameter_data, result_data, workflow_id, run_id, step_id)
            return to_return
    return {'Channel not found'}


@router.get("/return_partial_autocorrelation", tags=["return_partial_autocorrelation"])
# Validation is done inline in the input of the function
async def return_partial_autocorrelation(workflow_id: str, step_id: str, run_id: str,
                                         input_name: str,
                                         input_method: str | None = Query("none",
                                                                          regex="^(none)$|^(yw)$|^(ywadjusted)$|^(ywm)$|^(ywmle)$|^(ols)$|^(ols-inefficient)$|^(ols-adjusted)$|^(ld)$|^(ldadjusted)$|^(ldb)$|^(ldbiased)$|^(burg)$"),
                                         input_alpha: float | None = None, input_nlags: int | None = None,
                                         file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                         ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    # path_to_storage = get_local_storage_path(run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    # data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            z = pacf(raw_data[i], method=input_method, alpha=input_alpha, nlags=input_nlags)

            to_return = {
                'values_partial_autocorrelation': None,
                'confint': None
            }

            # Parsing the results of acf into a single object
            # Results will change depending on our input
            if input_alpha:
                to_return['values_partial_autocorrelation'] = z[0].tolist()
                to_return['confint'] = z[1].tolist()
            else:
                to_return['values_partial_autocorrelation'] = z.tolist()

            print("RETURNING VALUES")
            print(to_return)

            # Prepare the data to be written to the config file
            parameter_data = {
                'name': input_name,
                'method': input_method,
                'alpha': input_alpha,
                'nlags': input_nlags,
            }
            result_data = {
                'data_values_partial_autocorrelation': to_return['values_partial_autocorrelation'],
                'data_confint': to_return['confint'],
            }

            write_function_data_to_config_file(parameter_data, result_data, workflow_id, run_id, step_id)
            return to_return
    return {'Channel not found'}


@router.get("/return_filters", tags=["return_filters"])
# Validation is done inline in the input of the function besides
async def return_filters(
                         workflow_id: str, step_id: str, run_id: str,
                         input_name: str,
                         input_cutoff_1: int,
                         input_order: int,
                         input_fs: float,
                         input_cutoff_2: int | None = None,
                         input_analog: bool | None = False,
                         input_btype: str | None = Query("lowpass",
                                                         regex="^(lowpass)$|^(highpass)$|^(bandpass)$|^(bandstop)$"),
                         input_output: str | None = Query("ba", regex="^(ba)$|^(zpk)$|^(sos)$"),
                         input_worn: int | None = 512,
                         input_whole: bool | None = False,
                         input_fs_freq: float | None = None,
                         ) -> dict:
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    # Getting data from file
    # This should chnge in the future to use received data
    raw_data = data.get_data()
    # Get channgel of the eeg
    channels = data.ch_names
    info = data.info
    for i in range(len(channels)):
        if input_name == channels[i]:
            print("-------Starting Filters---------")
            print(input_fs_freq)
            print(input_whole)
            data_1 = raw_data[i]
            wn = None
            if input_btype != 'bandpass' and input_btype != 'bandstop':
                nyq = 0.5 * input_fs
                wn = input_cutoff_1 / nyq
            else:
                nyq = 0.5 * input_fs
                low = input_cutoff_1 / nyq
                high = input_cutoff_2 / nyq
                wn = [low, high]

            filter_created = butter(N=input_order, Wn=wn, btype=input_btype, analog=input_analog, output=input_output,
                                    fs=input_fs)

            filter_output = None
            frequency_response_output = None
            if input_output == "ba":
                filter_output = lfilter(filter_created[0], filter_created[1], raw_data)
                frequency_response_output = freqs(b=filter_created[0], a=filter_created[1], worN=input_worn)
            elif input_output == "zpk":
                frequency_response_output = freqs_zpk(z=filter_created[0], p=filter_created[1], k=filter_created[2],
                                                      worN=input_worn)
            elif input_output == "sos":
                # Must be searched and fixed
                filter_output = sosfilt(sos=filter_created, x=data_1, axis=-1, zi=None)
                if input_fs_freq:
                    frequency_response_output = sosfreqz(filter_created, worN=input_worn, whole=input_whole,
                                                         fs=input_fs_freq)
                else:
                    frequency_response_output = sosfreqz(filter_created, worN=input_worn, whole=input_whole)
            else:
                return {"error"}

            # print("-----------------------------------------------------------")
            # print("frequency_response_output is:")
            # print(frequency_response_output)
            # # print(type(frequency_response_output[1].tolist()))
            # # print(frequency_response_output[1].tolist())
            # #
            # print("filter_output is: ")
            # print(filter_output.tolist())
            to_return = {}
            to_return["frequency_w"] = frequency_response_output[0].tolist()
            # Since frequency h numbers are complex we convert them to strings
            temp_complex_freq = frequency_response_output[1].tolist()
            for complex_out_it, complex_out_val in enumerate(temp_complex_freq):
                temp_complex_freq[complex_out_it] = str(complex_out_val)
                # complex_out_it = str(complex_out_it)

            # print("IM HERE")
            to_return["frequency_h"] = temp_complex_freq

            # zpk doesnt have filter
            #
            if input_output == "ba":
                temp_filter_output = filter_output[0].tolist()
                for filter_out_it, filter_out_val in enumerate(temp_filter_output):
                    if math.isnan(filter_out_val):
                        temp_filter_output[filter_out_it] = None

                    if math.isinf(filter_out_val):
                        temp_filter_output[filter_out_it] = None

                to_return["filter"] = temp_filter_output
            elif input_output == "sos":
                to_return["filter"] = filter_output.tolist()

            print("RESULTS TO RETURN IS")
            print(to_return)

            return to_return


# Estimate welch
@router.get("/return_welch", tags=["return_welch"])
# TODO Create plot
# Validation is done inline in the input of the function
async def estimate_welch(
                        workflow_id: str, step_id: str, run_id: str,
                        input_name: str,
                         tmin: float | None = 0,
                         tmax: float | None = None,
                         input_window: str | None = Query("hann",
                                                          regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                         input_nperseg: int | None = 256,
                         input_noverlap: int | None = None,
                         input_nfft: int | None = 256,
                         input_return_onesided: bool | None = True,
                         input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                         input_axis: int | None = -1,
                         input_average: str | None = Query("mean", regex="^(mean)$|^(median)$"),
                         file_used: str | None = Query("original", regex="^(original)$|^(printed)$") ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    # data.crop(tmin=tmin, tmax=tmax)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    print("--------DATA----")
    print(raw_data)
    for i in range(len(channels)):
        if input_name == channels[i]:
            if input_window == "hann":
                f, pxx_den = signal.welch(raw_data[i], info['sfreq'], window=input_window,
                                          noverlap=input_noverlap, nperseg=input_nperseg, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)
            else:
                f, pxx_den = signal.welch(raw_data[i], info['sfreq'],
                                          window=signal.get_window(input_window, input_nperseg),
                                          noverlap=input_noverlap, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)
            plt.figure("psd welch")
            plt.semilogy(f, pxx_den)

            # plt.ylim([0.5e-3, 1])

            plt.xlabel('frequency [Hz]')

            plt.ylabel('PSD [V**2/Hz]')
            plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'welch_plot.png')

            plt.show()

            plt.clf()
            plt.close()

            to_return = {
                "frequencies": f.tolist(),
                "power spectral density": pxx_den.tolist()
            }

            # Prepare the data to be written to the config file
            parameter_data = {
                "window": input_window,
                "nperseg": input_nperseg,
                "noverlap": input_noverlap,
                "nfft": input_nfft,
                "return_onesided": input_return_onesided,
                "scaling": input_scaling,
                "axis": input_axis,
                "average": input_average,
            }

            result_data = {
                "data_frequencies": to_return["frequencies"],
                "data_power spectral density": to_return["power spectral density"]
            }

            write_function_data_to_config_file(workflow_id, step_id, run_id, parameter_data, result_data)

            return to_return
    return {'Channel not found'}

@router.get("/return_stft", tags=["return_stft"])
# Validation is done inline in the input of the function
async def estimate_stft(
                        workflow_id: str, step_id: str, run_id: str,
                        input_name: str,
                         tmin: float | None = 0,
                         tmax: float | None = None,
                         input_window: str | None = Query("hann",
                                                          regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                         input_nperseg: int | None = 256,
                         input_noverlap: int | None = None,
                         input_nfft: int | None = 256,
                         input_return_onesided: bool | None = True,
                         input_boundary: str | None = Query("zeros",
                                                          regex="^(zeros)$|^(even)$|^(odd)$|^(constant)$|^(None)$"),
                         input_padded: bool | None = True,
                         input_axis: int | None = -1,
                         file_used: str | None = Query("original", regex="^(original)$|^(printed)$")) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)


    # data.crop(tmin=tmin, tmax=tmax)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    if input_boundary == "None":
        input_boundary = None
    for i in range(len(channels)):
        if input_name == channels[i]:
            to_return = {}
            f, t, zxx_den = signal.stft(raw_data[i], info['sfreq'],
                                          window=input_window, nperseg=input_nperseg,
                                          noverlap=input_noverlap, nfft=input_nfft,
                                          return_onesided=input_return_onesided, boundary=input_boundary, padded=input_padded,
                                          axis=input_axis)
            # print(f'len zxx: {len(zxx_den.tolist())}')
            # print(f'len f:{len(f.tolist())}')
            # print(f'len t:{len(t.tolist())}')
            # for zxx_it, zxx in enumerate(zxx_den.tolist()):
            #     print(f'len {zxx_it} zxx: {len(zxx)}')
            # print(zxx)
            fig = plt.figure(figsize=(18, 12))
            amp = 2 * np.sqrt(2)
            plt.pcolormesh(t, f, np.abs(zxx_den), shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            # plt.show()

            # Convert the plot to HTML
            html_str = mpld3.fig_to_html(fig)
            to_return["figure"] = html_str

            # Save the plot to the local storage
            plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'stft_plot.png')

            # Prepare the data to be written to the config file
            parameter_data = {
                "window": input_window,
                "nperseg": input_nperseg,
                "noverlap": input_noverlap,
                "nfft": input_nfft,
                "return_onesided": input_return_onesided,
                "boundary": input_boundary,
                "padded": input_padded,
                "axis": input_axis,
            }

            result_data = {
                "path_stft_figure": "plot.png",
            }

            write_function_data_to_config_file(workflow_id, step_id, run_id, parameter_data, result_data)
            return to_return
    return {'Channel not found'}


# Find peaks
@router.get("/return_peaks", tags=["return_peaks"])
# Validation is done inline in the input of the function
async def return_peaks(workflow_id: str, step_id: str, run_id: str,
                       input_name: str,
                       input_height=None,
                       input_threshold=None,
                       input_distance: int | None = None,
                       input_prominence=None,
                       input_width=None,
                       input_wlen: int | None = None,
                       input_rel_height: float | None = None,
                       input_plateau_size=None,
                       file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                       ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    # ss value should be a valid integer
    raw_data = data.get_data(return_times=True)
    channels = data.ch_names

    print(input_height)
    validated_data = validate_and_convert_peaks(input_height, input_threshold, input_prominence, input_width,
                                                input_plateau_size)

    print("--------DATA INFO----")
    print(data.ch_names)
    print(data.info)
    # print(data.info.meas_date)
    print(data)
    print(data.info["meas_date"])
    # print("--------VALIDATED----")
    # print(input_height)
    # print(type(validated_data["width"]))
    # print(validated_data)
    for i in range(len(channels)):
        if input_name == channels[i]:

            find_peaks_result = signal.find_peaks(x=raw_data[0][i], height=validated_data["height"],
                                                  threshold=validated_data["threshold"],
                                                  distance=input_distance, prominence=validated_data["prominence"],
                                                  width=validated_data["width"], wlen=input_wlen,
                                                  rel_height=input_rel_height,
                                                  plateau_size=validated_data["plateau_size"])
            print("--------RESULTS----")
            print(find_peaks_result)
            # print(_)n
            to_return = {
                "signal": None,
                "signal_time": None,
                "start_date_time": None,
                "peaks": None,
                "peak_heights": None,
                "left_thresholds": None,
                "right_thresholds": None,
                "prominences": None,
                "left_bases": None,
                "right_bases": None,
                "width_heights": None,
                "left_ips": None,
                "right_ips": None,
                "left_edges": None,
                "right_edges": None,
                "plateau_sizes": None,
            }
            to_return["signal"] = raw_data[0][i].tolist()
            to_return["signal_time"] = raw_data[1].tolist()
            to_return["start_date_time"] = data.info["meas_date"].timestamp()
            # to_return["start_date_time"] = json.dumps(data.info["meas_date"], default=datetime_handler)

            # print("raw data type")
            # print(type(raw_data[0][i].tolist()))
            # print("raw data time type")
            # print(type(raw_data[1].tolist()))
            # print(raw_data[1].tolist())
            to_return["peaks"] = find_peaks_result[0].tolist()

            if input_height:
                to_return["peak_heights"] = find_peaks_result[1]["peak_heights"].tolist()

            if input_threshold:
                to_return["left_thresholds"] = find_peaks_result[1]["left_thresholds"].tolist()
                to_return["right_thresholds"] = find_peaks_result[1]["right_thresholds"].tolist()

            if input_prominence:
                to_return["prominences"] = find_peaks_result[1]["prominences"].tolist()
                to_return["right_bases"] = find_peaks_result[1]["right_bases"].tolist()
                to_return["left_bases"] = find_peaks_result[1]["left_bases"].tolist()

            if input_width:
                to_return["width_heights"] = find_peaks_result[1]["width_heights"].tolist()
                to_return["left_ips"] = find_peaks_result[1]["left_ips"].tolist()
                to_return["right_ips"] = find_peaks_result[1]["right_ips"].tolist()

            if input_plateau_size:
                to_return["plateau_sizes"] = find_peaks_result[1]["plateau_sizes"].tolist()
                to_return["left_edges"] = find_peaks_result[1]["left_edges"].tolist()
                to_return["right_edges"] = find_peaks_result[1]["right_edges"].tolist()

            fig = plt.figure(figsize=(18, 12))
            border = np.sin(np.linspace(0, 3 * np.pi, raw_data[0][i].size))
            plt.plot(raw_data[0][i])
            plt.plot(find_peaks_result[0].tolist(), raw_data[0][i][find_peaks_result[0].tolist()], "x")

            if input_prominence:
                plt.vlines(x=find_peaks_result[0].tolist(),
                           ymin=raw_data[0][i][find_peaks_result[0].tolist()] - find_peaks_result[1][
                               "prominences"].tolist(),
                           ymax=raw_data[0][i][find_peaks_result[0].tolist()], color="C1")

            if input_width:
                plt.hlines(y=find_peaks_result[1]["width_heights"].tolist(),
                           xmin=find_peaks_result[1]["left_ips"].tolist(),
                           xmax=find_peaks_result[1]["right_ips"].tolist(), color="C1")
            # plt.plot(find_peaks_result, "x")
            # plt.plot(find_peaks_result, raw_data[i][find_peaks_result], "x")

            # plt.plot(np.zeros_like(x), "--", color="gray")
            plt.plot(np.zeros_like(raw_data[0][i]), "--", color="red")
            plt.show()

            # Convert plot to html
            html_str = mpld3.fig_to_html(fig)

            # Save plot to local storage
            plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'plot.png')

            to_return["figure"] = html_str

            # Prepare the data to be written to the config file
            parameter_data = {
                "name": input_name,
                "height": input_height,
                "threshold": input_threshold,
                "distance": input_distance,
                "prominence": input_prominence,
                "width": input_width,
                "wlen": input_wlen,
                "rel_height": input_rel_height,
                "plateau_size": input_plateau_size,
            }

            result_data = {
                "path_peaks_plot" : "plot.png",
            }

            write_function_data_to_config_file(workflow_id, step_id, run_id, parameter_data, result_data)

            return to_return
    return {'Channel not found'}


# Estimate welch
@router.get("/return_periodogram", tags=["return_periodogram"])
# Validation is done inline in the input of the function
# TODO Create plot

async def estimate_periodogram(workflow_id: str, step_id: str, run_id: str,input_name: str,
                               tmin: float | None = 0,
                               tmax: float | None = None,
                               input_window: str | None = Query("hann",
                                                                regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                               input_nfft: int | None = 256,
                               input_return_onesided: bool | None = True,
                               input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                               input_axis: int | None = -1,
                               file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                               ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)


    # data.crop(tmin=tmin, tmax=tmax)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            f, pxx_den = signal.periodogram(raw_data[i], info['sfreq'], window=input_window,
                                            nfft=input_nfft, return_onesided=input_return_onesided,
                                            scaling=input_scaling,
                                            axis=input_axis)

            plt.figure("psd periodogram")

            plt.semilogy(f, pxx_den)
            # plt.ylim([1e-7, 1e2])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz]')
            plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'periodogram_plot.png')
            plt.show()
            plt.clf()
            plt.close()

            return {'frequencies': f.tolist(), 'power spectral density': pxx_den.tolist()}
    return {'Channel not found'}

# @router.get("/discrete_wavelet_transform", tags=["discrete_wavelet_transform"])
# # Validation is done inline in the input of the function
# async def discrete_wavelet_transform(input_name: str,
#                                      wavelet: str |None = Query("db1",
#                                                                 regex="^(db1)$|^(db2)$|^(coif1)$|^(coif2)$"),
#                                      mode: str | None = Query("sym",
#                                                               regex="^(sym)$|^(zpd)$|^(cpd)$|^(ppd)$|^(sp1)$|^(per)$"),
#                                      level: int | None = None) -> dict:
#     raw_data = data.get_data()
#     info = data.info
#     channels = data.ch_names
#     for i in range(len(channels)):
#         if input_name == channels[i]:
#             if level!=None:
#                 coeffs = pywt.wavedec(raw_data[i], wavelet=wavelet, mode=mode, level=level)
#             else:
#                 w = pywt.Wavelet(str(wavelet))
#                 level = pywt.dwt_max_level(data_len=np.shape(raw_data[i])[0], filter_len=w.dec_len)
#                 coeffs = pywt.wavedec(raw_data[i], wavelet=wavelet, mode=mode, level=level)
#             return {'coefficients': coeffs}
#     return {'Channel not found'}

# Return power_spectral_density
@router.get("/return_power_spectral_density", tags=["return_power_spectral_density"])
# Validation is done inline in the input of the function
# TODO Create plot
# TODO TMIN and TMAX probably should be removed
async def return_power_spectral_density(workflow_id: str,
                                        step_id: str,
                                        run_id: str,
                                        input_name: str,
                                        # tmin: float | None = None,
                                        # tmax: float | None = None,
                                        input_fmin: float | None = 0,
                                        input_fmax: float | None = None,
                                        input_bandwidth: float | None = None,
                                        input_adaptive: bool | None = False,
                                        input_low_bias: bool | None = True,
                                        input_normalization: str | None = "length",
                                        input_output: str | None = "power",
                                        input_n_jobs: int | None = 1,
                                        input_verbose: str | None = None,
                                        file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                        ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)


    # data.crop(tmin=tmin, tmax=tmax)
    raw_data = data.get_data()
    info = data.info

    channels = data.ch_names
    # print(input_height)
    validated_input_verbose = validate_and_convert_power_spectral_density(input_verbose)

    print("--------VALIDATED----")
    print(validated_input_verbose)

    for i in range(len(channels)):
        if input_name == channels[i]:
            # Verbose is always none cause it might not be needed
            # Output is always power because alternative might not be needed
            psd_results, freqs = psd_array_multitaper(x=raw_data[i],
                                                      sfreq=info['sfreq'],
                                                      fmin=input_fmin,
                                                      fmax=input_fmax,
                                                      bandwidth=input_bandwidth,
                                                      adaptive=input_adaptive,
                                                      low_bias=input_low_bias,
                                                      normalization=input_normalization,
                                                      output=input_output,
                                                      n_jobs=input_n_jobs,
                                                      verbose=None
                                                      )
            print("--------PSD----")
            print(psd_results)
            print(freqs)
            plt.figure("psd multitaper")

            plt.semilogy(freqs, psd_results)

            # plt.ylim([0.5e-3, 1])

            plt.xlabel('frequency [Hz]')

            plt.ylabel('PSD [V**2/Hz]')
            plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'multitaper_plot.png')

            plt.show()
            plt.clf()
            plt.close()

            to_return = {'frequencies': freqs.tolist(), 'power spectral density': psd_results.tolist()}
            return to_return
    return {'Channel not found'}

@router.get("/calculate_SpO2")
async def SpO2_Hypothesis():
    signals, signal_headers, header = highlevel.read_edf('NIA test.edf')
    for i in range(len(signal_headers)):
        if "SpO2" in signal_headers[i]['label']:
            modified_array = np.delete(signals[i], np.where(signals[i] == 0))
            if modified_array != []:
                minimum_SpO2 = np.min(modified_array)
                number_of_samples = np.shape(np.where(modified_array < 92))[1]
                time_in_seconds = number_of_samples / signal_headers[i]['sample_frequency']
                return {'minimumSpO2': minimum_SpO2, 'time':time_in_seconds}
            else:
                return {"All values are 0"}
    return {'Channel not found'}

@router.get("/return_alpha_delta_ratio", tags=["return_alpha_delta_ratio"])
async def calculate_alpha_delta_ratio(workflow_id: str, step_id: str, run_id: str,input_name: str,
                                      tmin: float | None = 0,
                                      tmax: float | None = None,
                                      input_window: str | None = Query("hann",
                                                          regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                                      input_nperseg: int | None = 256,
                                      input_noverlap: int | None = None,
                                      input_nfft: int | None = None,
                                      input_return_onesided: bool | None = True,
                                      input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                                      input_axis: int | None = -1,
                                      input_average: str | None = Query("mean", regex="^(mean)$|^(median)$"),
                                      file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                      ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)


    # data.crop(tmin=tmin, tmax=tmax)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            if input_window == "hann":
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'], window=input_window,
                                          noverlap=input_noverlap, nperseg=input_nperseg, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)
            else:
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'],
                                          window=signal.get_window(input_window, input_nperseg),
                                          noverlap=input_noverlap, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)

            list_power = []
            peak_f = []
            # Define alpha lower and upper limits
            low, high = 8, 13

            # Find intersecting values in frequency vector
            idx_alpha = np.logical_and(freqs >= low, freqs <= high)
            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

            # Compute the absolute power by approximating the area under the curve
            alpha_power = simps(psd[idx_alpha], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f<=high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_alpha])])

            #################################################

            # delta power
            low, high = 0.5, 3.9

            # Find intersecting values in frequency vector
            idx_05_4 = np.logical_and(freqs >= low, freqs <= high)

            # Compute the absolute power by approximating the area under the curve
            delta_power = simps(psd[idx_05_4], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f <= high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_alpha])])

            # theta power
            low, high = 4, 8

            # Find intersecting values in frequency vector
            idx_05_4 = np.logical_and(freqs >= low, freqs < high)

            # Compute the absolute power by approximating the area under the curve
            theta_power = simps(psd[idx_05_4], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f <= high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_alpha])])

            # beta power
            low, high = 13, 30

            # Find intersecting values in frequency vector
            idx_05_4 = np.logical_and(freqs > low, freqs <= high)

            # Compute the absolute power by approximating the area under the curve
            beta_power = simps(psd[idx_05_4], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f <= high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_alpha])])

            list_power.append(beta_power)
            list_power.append(theta_power)
            list_power.append(alpha_power)
            list_power.append(delta_power)
            names_power = ['Beta', 'Theta', 'Alpha','Delta']
            df_names = pd.DataFrame(names_power, columns=['Band'])
            df_power = pd.DataFrame(list_power, columns=['Power (uV^2)'])
            peak_f_new = []
            peak_f_new.append(peak_f[3])
            peak_f_new.append(peak_f[2])
            peak_f_new.append(peak_f[0])
            peak_f_new.append(peak_f[1])
            df_peak = pd.DataFrame(peak_f_new, columns=['Peak (Hz)'])

            df = pd.concat([df_names, df_power, df_peak],1)

            df['index'] = df.index
            return {'alpha_delta_ratio': alpha_power/delta_power, 'alpha_delta_ratio_df': df.to_json(orient='records')}

@router.get("/return_alpha_delta_ratio_periodogram", tags=["return_alpha_delta_ratio_periodogram"])
async def calculate_alpha_delta_ratio_periodogram(workflow_id: str,
                                                  step_id: str,
                                                  run_id: str,
                                                  input_name: str,
                                                  tmin: float | None = 0,
                                                  tmax: float | None = None,
                                                  input_window: str | None = Query("hann",
                                                                                   regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                                                  input_nfft: int | None = None,
                                                  input_return_onesided: bool | None = True,
                                                  input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                                                  input_axis: int | None = -1,
                                                  file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                                  ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    # data.crop(tmin=tmin, tmax=tmax)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            if input_window == "hann":
                freqs, psd = signal.periodogram(raw_data[i]*(10**3), info['sfreq'], window=input_window,
                                                nfft=input_nfft,return_onesided=input_return_onesided,
                                                scaling=input_scaling, axis=input_axis)
            else:
                freqs, psd = signal.periodogram(raw_data[i]*(10**3), info['sfreq'],
                                                window=signal.get_window(input_window),
                                                nfft=input_nfft, return_onesided=input_return_onesided, scaling=input_scaling,
                                                axis=input_axis)

            list_power = []
            peak_f = []

            # beta power
            low, high = 13, 30
            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

            # Find intersecting values in frequency vector
            idx_05_4 = np.logical_and(freqs > low, freqs <= high)

            # Compute the absolute power by approximating the area under the curve
            beta_power = simps(psd[idx_05_4], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f <= high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_05_4])])

            # theta power
            low, high = 4, 8

            # Find intersecting values in frequency vector
            idx_05_4 = np.logical_and(freqs >= low, freqs < high)

            # Compute the absolute power by approximating the area under the curve
            theta_power = simps(psd[idx_05_4], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f <= high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_05_4])])

            # Define alpha lower and upper limits
            low, high = 8, 13

            # Find intersecting values in frequency vector
            idx_alpha = np.logical_and(freqs >= low, freqs <= high)
            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

            # Compute the absolute power by approximating the area under the curve
            alpha_power = simps(psd[idx_alpha], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f<=high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_alpha])])

            #################################################

            # delta power
            low, high = 0.5, 3.9

            # Find intersecting values in frequency vector
            idx_05_4 = np.logical_and(freqs >= low, freqs <= high)

            # Compute the absolute power by approximating the area under the curve
            delta_power = simps(psd[idx_05_4], dx=freq_res)
            new_freqs = []
            for f in freqs:
                if f >= low and f <= high:
                    new_freqs.append(f)

            peak_f.append(new_freqs[np.argmax(psd[idx_05_4])])


            list_power.append(beta_power)
            list_power.append(theta_power)
            list_power.append(alpha_power)
            list_power.append(delta_power)
            names_power = ['Beta', 'Theta', 'Alpha','Delta']
            df_names = pd.DataFrame(names_power, columns=['Band'])
            df_power = pd.DataFrame(list_power, columns=['Power (uV^2)'])
            df_peak = pd.DataFrame(peak_f, columns=['Peak (Hz)'])

            df = pd.concat([df_names, df_power, df_peak],1)
            print(df)

            df['index'] = df.index
            return {'alpha_delta_ratio': alpha_power/delta_power, 'alpha_delta_ratio_df': df.to_json(orient='records')}


@router.get("/return_asymmetry_indices", tags=["return_asymmetry_indices"])
async def calculate_asymmetry_indices(workflow_id: str, step_id: str, run_id: str,
                                      input_name_1: str,
                                      input_name_2: str,
                                      input_window: str | None = Query("hann",
                                                          regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                                      input_nperseg: int | None = 256,
                                      input_noverlap: int | None = None,
                                      input_nfft: int | None = None,
                                      input_return_onesided: bool | None = True,
                                      input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                                      input_axis: int | None = -1,
                                      input_average: str | None = Query("mean", regex="^(mean)$|^(median)$"),
                                      file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                      ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)


    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name_1 == channels[i]:
            if input_window == "hann":
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'], window=input_window,
                                          noverlap=input_noverlap, nperseg=input_nperseg, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)
            else:
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'],
                                          window=signal.get_window(input_window, input_nperseg),
                                          noverlap=input_noverlap, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)

            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

            # Compute the absolute power by approximating the area under the curve
            abs_power_1 = simps(psd, dx=freq_res)
        elif input_name_2 == channels[i]:
            if input_window == "hann":
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'], window=input_window,
                                          noverlap=input_noverlap, nperseg=input_nperseg, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)
            else:
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'],
                                          window=signal.get_window(input_window, input_nperseg),
                                          noverlap=input_noverlap, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)

            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

            # Compute the absolute power by approximating the area under the curve
            abs_power_2 = simps(psd, dx=freq_res)
            print(abs_power_2)


    asymmetry_index = (np.log(abs_power_1) - np.log(abs_power_2))/(np.log(abs_power_1) + np.log(abs_power_2))

    return {'asymmetry_indices': asymmetry_index}

@router.get("/return_alpha_variability", tags=["return_alpha_variability"])
async def calculate_alpha_variability(workflow_id: str,
                                      step_id: str,
                                      run_id: str,
                                      input_name: str,
                                      tmin: float | None = 0,
                                      tmax: float | None = None,
                                      input_window: str | None = Query("hann",
                                                          regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                                      input_nperseg: int | None = 256,
                                      input_noverlap: int | None = None,
                                      input_nfft: int | None = None,
                                      input_return_onesided: bool | None = True,
                                      input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                                      input_axis: int | None = -1,
                                      input_average: str | None = Query("mean", regex="^(mean)$|^(median)$"),
                                      file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                      ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)


    # data.crop(tmin=tmin, tmax=tmax)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            if input_window == "hann":
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'], window=input_window,
                                          noverlap=input_noverlap, nperseg=input_nperseg, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)
            else:
                freqs, psd = signal.welch(raw_data[i]*(10**3), info['sfreq'],
                                          window=signal.get_window(input_window, input_nperseg),
                                          noverlap=input_noverlap, nfft=input_nfft,
                                          return_onesided=input_return_onesided, scaling=input_scaling,
                                          axis=input_axis, average=input_average)
            # Define alpha lower and upper limits
            low, high = 8, 13

            # Find intersecting values in frequency vector
            idx_alpha = np.logical_and(freqs >= low, freqs <= high)
            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

            # Compute the absolute power by approximating the area under the curve
            alpha_power = simps(psd[idx_alpha], dx=freq_res)
            #################################################

            #
            low, high = 1, 20

            # Find intersecting values in frequency vector
            idx_1_20 = np.logical_and(freqs >= low, freqs <= high)

            # Compute the absolute power by approximating the area under the curve
            total_power = simps(psd[idx_1_20], dx=freq_res)

            return {'alpha_variability': alpha_power/total_power}

@router.get("/return_predictions", tags=["return_predictions"])
async def return_predictions(workflow_id: str, step_id: str, run_id: str,input_name: str,
                             input_test_size: int,
                             input_future_seconds: int,
                             input_start_p: int | None = 1,
                             input_start_q: int | None = 1,
                             input_max_p: int | None = 5,
                             input_max_q: int | None = 5,
                             input_method: str | None = Query("lbfgs",
                                                              regex="^(lbfgs)$|^(newton)$|^(nm)$|^(bfgs)$|^(powell)$|^(cg)$|^(ncg)$|^(basinhopping)$"),
                             input_information_criterion: str | None = Query("aic",
                                                                             regex="^(aic)$|^(bic)$|^(hqic)$|^(oob)$"),
                             file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                             ):
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    raw_data = data.get_data()
    channels = data.ch_names
    info = data.info
    sampling_frequency = info['sfreq']
    for i in range(len(channels)):
        if input_name == channels[i]:
            data_channel = raw_data[i]
            train, test = data_channel[:-input_test_size], data_channel[-input_test_size:]
            #x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], data_channel.shape[0]))
            model = auto_arima(train, start_p=input_start_p, start_q=input_start_q,
                               test='adf',
                               max_p=input_max_p, max_q=input_max_q,
                               m=1,
                               d=1,
                               seasonal=False,
                               start_P=0,
                               D=None,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True,
                               method=input_method,
                               information_criterion=input_information_criterion)
            prediction, confint = model.predict(n_periods=input_test_size, return_conf_int=True)
            smape = calcsmape(test, prediction)
            example = model.summary()
            results_as_html_1 = example.tables[0].as_html()
            print('html')
            print(results_as_html_1)
            df_0 = pd.read_html(results_as_html_1, header=0, index_col=0)[0]
            print('json')
            print(df_0.to_json(orient="split"))



            results_as_html_2 = example.tables[1].as_html()
            df_1 = pd.read_html(results_as_html_2, header=0, index_col=0)[0]

            results_as_html_3 = example.tables[2].as_html()
            df_2 = pd.read_html(results_as_html_3, header=0, index_col=0)[0]

            z = input_future_seconds * sampling_frequency

            prediction, confint = model.predict(n_periods=int(z), return_conf_int=True)
            return {'predictions': prediction.tolist(), 'error': smape, 'confint': confint, 'first_table':results_as_html_1, 'second_table':results_as_html_2, 'third_table':results_as_html_3}
    return {'Channel not found'}

@router.get("/sleep_stage_classification", tags=["sleep_stage_classification"])
async def sleep_stage_classify(workflow_id: str,
                               step_id: str,
                               run_id: str,
                               eeg_chanel_name: str,
                               eog_channel_name: str | None = Query(default=None),
                               emg_channel_name: str | None = Query(default=None),
                               file_used: str | None = Query("original", regex="^(original)$|^(printed)$")):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    sls = SleepStaging(data, eeg_name=eeg_chanel_name, eog_name=eog_channel_name, emg_name=emg_channel_name)

    y_pred = sls.predict()

    df = sls.predict_proba()

    confidence = sls.predict_proba().max(1)

    df_pred = pd.DataFrame({'Stage': y_pred, 'Confidence': confidence})

    df.to_csv(path_to_storage + '/output/sleep_stage.csv', index=False)
    df_pred.to_csv(path_to_storage + '/output/sleep_stage_confidence.csv', index=False)

    # Convert to format of general sleep stage and save new file to interim storage and output
    converted_df = convert_yasa_sleep_stage_to_general(path_to_storage + '/output/sleep_stage_confidence.csv')
    # converted_df.to_csv(path_to_storage + '/neurodesk_interim_storage/new_hypnogram.csv', index=False)
    converted_df.to_csv(path_to_storage + '/output/new_hypnogram.csv', index=False)

    # Convert to annotation format incase user wants to use in manual scoring
    convert_generic_sleep_score_to_annotation("output/new_hypnogram.csv", workflow_id, run_id,  step_id)

    # Convert and send to frontend
    df['id'] = df.index
    df_pred['id'] = df_pred.index
    return {'sleep_stage': df.to_json(orient='records'), # Predicted probability for each sleep stage for each 30-sec epoch of data
            'sleep_stage_confidence': df_pred.to_json(orient='records')} # dataframe with the predicted stages and confidence

# Spindles detection
@router.get("/spindles_detection")
async def detect_spindles(
                          workflow_id: str,
                          step_id: str,
                          run_id: str,
                          name: str,
                          freq_sp_low: float | None = 12,
                          freq_sp_high: float | None = 15,
                          freq_broad_low: float | None = 1,
                          freq_broad_high: float | None = 30,
                          duration_low: float | None = 0.5,
                          duration_high: float | None = 2,
                          min_distance: float | None = 500,
                          rel_pow: float | None = None,
                          corr: float | None = None,
                          rms: float | None = None,
                          multi_only: bool | None = False,
                          remove_outliers: bool | None = False,
                          file_used: str | None = Query("original", regex="^(original)$|^(printed)$")):
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)


    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    list_all = []
    for i in range(len(channels)):
        if name == channels[i]:
            sp = spindles_detect(data=raw_data[i] * 1e6,
                                 sf=info['sfreq'],
                                 # hypno=,
                                 # include=,
                                 freq_sp= (freq_sp_low, freq_sp_high),
                                 freq_broad= (freq_broad_low, freq_broad_high),
                                 duration= (duration_low, duration_high),
                                 min_distance= min_distance,
                                 thresh={'rel_pow': rel_pow, 'corr': corr, 'rms': rms},
                                 multi_only=multi_only,
                                 remove_outliers=remove_outliers
                                 )
            to_return ={}
            fig = plt.figure(figsize=(18, 12))
            plt.plot(raw_data[0][i])
            html_str =mpld3.fig_to_html(fig)
            to_return["figure"] = html_str
            if sp==None:
                to_return["detected"] = "No Spindles"
                return to_return
            else:
                df_sync = sp.get_sync_events(center="Peak", time_before=0.4, time_after=0.8)
                fig, ax = plt.subplots(1, 1, figsize=(9, 6))
                seaborn.lineplot(data=df_sync, x='Time', y='Amplitude', hue='Channel', palette="plasma", ci=95, ax=ax)
                # ax.legend(frameon=False, loc='lower right')
                ax.set_xlim(df_sync['Time'].min(), df_sync['Time'].max())
                ax.set_title('Spindles Average')
                ax.set_xlabel('Time (sec)')
                ax.set_ylabel('Amplitude (uV)')
                plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'plot.png')

                df = sp.summary()
                for i in range(len(df)):
                    list_start_end = []
                    start = df.iloc[i]['Start'] * info['sfreq']
                    end = df.iloc[i]['End'] * info['sfreq']
                    list_start_end.append(start)
                    list_start_end.append(end)
                    list_all.append(list_start_end)

                    to_return["detected spindles"] = list_all
                return to_return
    return {'Channel not found'}


@router.get("/return_available_hypnograms", tags=["return_available_hypnograms"])
async def return_available_hypnograms(workflow_id: str,
                          step_id: str,
                          run_id: str,):
    """This functions shows all hypnograms created from automatic and stored in interim storage in
    an auto sleep scoring function when redirected to manual sleep sccoring"""
    path = get_local_neurodesk_storage_path(workflow_id, run_id, step_id)
    list_of_files = os.listdir(path)
    return { "available_hypnograms": list_of_files}


@router.get("/initialise_hypnograms", tags=["initialise_hypnograms"])
async def initialise_hypnograms(workflow_id: str,
                          step_id: str,
                          run_id: str,):
    """This functions transfers all hypnogram to interim storage in a new manul sleep scoring function
     run or autoscoring funciton"""
    path = get_local_storage_path(workflow_id, run_id, step_id)
    list_of_files = os.listdir(path)
    list_of_copied_files = []
    for file in list_of_files:
        if file.endswith(".txt") or file.endswith(".csv"):
            shutil.copyfile(path + "/" + file, get_local_neurodesk_storage_path(workflow_id, run_id, step_id) + "/" + file)
            list_of_copied_files.append(file)

    return { "available_hypnograms": list_of_copied_files}


# Slow Waves detection
@router.get("/slow_waves_detection")
async def detect_slow_waves(
                          workflow_id: str,
                          step_id: str,
                          run_id: str,
                          name: str,
                          freq_sw_low: float | None = 12,
                          freq_sw_high: float | None = 15,
                          duration_negative_low: float | None = 0.5,
                          duration_negative_high: float | None = 2,
                          duration_positive_low: float | None = 0.5,
                          duration_positive_high: float | None = 2,
                          amplitude_positive_low: int | None = 0.5,
                          amplitude_positive_high: int | None = 2,
                          amplitude_negative_low: int | None = 0.5,
                          amplitude_negative_high: int | None = 2,
                          amplitude_ptp_low: int | None = 0.5,
                          amplitude_ptp_high: int | None = 2,
                          coupling: bool | None = False,
                          remove_outliers: bool | None = False,
                          file_used: str | None = Query("original", regex="^(original)$|^(printed)$")):
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    list_all = []
    for i in range(len(channels)):
        if name == channels[i]:
            print("gegegegege")
            print(channels[i])
            print(freq_sw_low)
            print(freq_sw_high)
            print(duration_negative_low)
            print(duration_negative_high)
            print(amplitude_positive_high)
            print(freq_sw_low)
            print(amplitude_ptp_low)
            print(remove_outliers)
            print(coupling)

            # SW_list = []

            sw = sw_detect(data =raw_data[i] * 1e6,
                           sf =info['sfreq'],
                           freq_sw= (freq_sw_low, freq_sw_high),
                           dur_neg=(duration_negative_low, duration_negative_high),
                           dur_pos=(duration_positive_low, duration_positive_high),
                           amp_neg=(amplitude_negative_low, amplitude_negative_high),
                           amp_pos=(amplitude_positive_low, amplitude_positive_high),
                           amp_ptp=(amplitude_ptp_low, amplitude_ptp_high),
                           coupling=coupling,
                           remove_outliers=remove_outliers
                           )


            if sw==None:
                return {'No slow waves'}
            else:
                df_sync = sw.get_sync_events(center="NegPeak", time_before=0.4, time_after=0.8)
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                seaborn.lineplot(data=df_sync, x='Time', y='Amplitude', hue='Channel', palette="plasma", ci=95, ax=ax)
                # ax.legend(frameon=False, loc='lower right')
                ax.set_xlim(df_sync['Time'].min(), df_sync['Time'].max())
                ax.set_title('Average SW')
                ax.set_xlabel('Time (sec)')
                ax.set_ylabel('Amplitude (uV)')
                plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'plot.png')

                df = sw.summary()
                for i in range(len(df)):
                    list_start_end = []
                    start = df.iloc[i]['Start'] * info['sfreq']
                    end = df.iloc[i]['End'] * info['sfreq']
                    list_start_end.append(start)
                    list_start_end.append(end)
                    list_all.append(list_start_end)
                return {'detected slow waves': list_all}
    return {'Channel not found'}

@router.get("/sleep_statistics_hypnogram")
async def sleep_statistics_hypnogram(
                                    workflow_id: str,
                                    step_id: str,
                                    run_id: str,
                                     sampling_frequency: float | None = Query(default=1/30)):
    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])

    df = pd.DataFrame.from_dict(sleep_statistics(list(hypno['stage']), sf_hyp=sampling_frequency), orient='index', columns=['value'])

    # print("DF Altered")
    # print(df)
    # print(df.T)
    # print(df.T.to_json(orient='records'))
    # print("DF original")
    # print(df.to_json(orient='records'))
    # print(df.to_json(orient='split'))
    df = df.T
    # df['index'] = df.index
    df.insert(0, 'id', range(1, 1 + len(df)))
    return{'sleep_statistics': df.to_json(orient='records')}

@router.get("/sleep_transition_matrix")
async def sleep_transition_matrix(workflow_id: str,
                                    step_id: str,
                                    run_id: str,):
    #fig = plt.figure(1)
    #ax = plt.subplot(111)

    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    to_return = {}
    plt.figure("sleep_transition_matrix")

    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])

    counts, probs = yasa.transition_matrix(list(hypno['stage']))

    # Start the plot
    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws,
                                    figsize=(5, 5))
    sns.heatmap(probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
                cbar_ax=cbar_ax, cmap='YlOrRd', annot=True, fmt='.2f',
                cbar_kws={"orientation": "horizontal", "fraction": 0.1,
                          "label": "Transition probability"})
    ax.set_xlabel("To sleep stage")
    ax.xaxis.tick_top()
    ax.set_ylabel("From sleep stage")
    ax.xaxis.set_label_position('top')
    # plt.show()
    #  Temporarilly saved in root directory should change to commented
    # fig.savefig( path_to_storage + "/output/" + 'sleep_transition_matrix.png')
    # plt.savefig(NeurodesktopStorageLocation + '/sleep_transition_matrix.png')
    plt.savefig(get_local_storage_path(workflow_id,run_id, step_id) + "/output/" + 'sleep_transition_matrix.png')

    # html_str = mpld3.fig_to_html(fig)
    # to_return["figure"] = html_str

    return{'counts_transition_matrix':counts.to_json(orient='split'),  # Counts transition matrix (number of transitions from stage A to stage B).
           'conditional_probability_transition_matrix':probs.to_json(orient='split'), # Conditional probability transition matrix, i.e. given that current state is A, what is the probability that the next state is B.
           'figure': to_return}

@router.get("/sleep_stability_extraction")
async def sleep_stability_extraction(workflow_id: str,
                                    step_id: str,
                                    run_id: str,):
    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])

    counts, probs = yasa.transition_matrix(list(hypno['stage']))

    return{'sleep_stage_stability': np.diag(probs.loc[2:, 2:]).mean().round(3)} # stability of sleep stages

# 2nd page
@router.get("/spectrogram_yasa")
async def spectrogram_yasa(
                           workflow_id: str,
                           step_id: str,
                           run_id: str,
                           name: str,
                           current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):

    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    data = mne.io.read_raw_fif(path_to_storage + "/" + files["edf"])
    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])


    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    print(channels)
    sf = info['sfreq']

    for i in range(len(channels)):
        if name == channels[i]:
            array_data = raw_data[i]
            hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)
            to_return = {}
            plt.figure("spectrogram_plot")
            yasa.plot_spectrogram(array_data, sf, hypno, cmap='Spectral_r')
            # plt.show()

            # html_str = mpld3.fig_to_html(fig)
            # to_return["figure"] = html_str
            #  Temporarilly saved in root directory should change to commented

            # fig.savefig(path_to_storage + "/output/" + 'spectrogram.png')
            # plt.savefig(NeurodesktopStorageLocation + '/spectrogram.png')
            plt.savefig(
                get_local_storage_path(workflow_id,run_id, step_id) + "/output/" + 'spectrogram.png')

            return {'figure': to_return}
    return {'Channel not found'}

@router.get("/bandpower_yasa")
async def bandpower_yasa(workflow_id: str,
                         step_id: str,
                         run_id: str,
                         relative: bool | None = False,
                         bandpass: bool | None = False,
                         include: list[int] | None = Query(default=[2,3]),
                         current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):
    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    data = mne.io.read_raw_fif(path_to_storage + "/" + files["edf"])
    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])

    # data = mne.io.read_raw_fif("example_data/XX_Firsthalf_raw.fif")
    # hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    print(channels)
    sf = info['sfreq']

    hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)
    print("include")
    print(include)
    df = yasa.bandpower(data, hypno=hypno, relative=relative, bandpass=bandpass, include=include)
    print(df)
    print('yesssss')
    df['Channel'] = df.index
    print(df)

    #Add index as column
    df['index'] = df.index
    return {'bandpower':df.to_json(orient='split')}

#  3rd page
@router.get("/spindles_detect_two_dataframes")
async def spindles_detect_two_dataframes(
                                         workflow_id: str,
                                         step_id: str,
                                         run_id: str,
                                         min_distance: int | None = Query(default=500),
                                         freq_sp: list[int] | None = Query(default=[12,15]),
                                         freq_broad: list[int] | None = Query(default=[1,30]),
                                         include: list[int] | None = Query(default=[2,3]),
                                         remove_outliers: bool | None = Query(default=False),
                                         current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):
    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    data = mne.io.read_raw_fif(path_to_storage + "/" + files["edf"])
    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])

    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    sf = info['sfreq']
    hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)

    sp = yasa.spindles_detect(raw_data, sf=sf, hypno=hypno, include=include, freq_sp=freq_sp, freq_broad=freq_broad,
                              min_distance=min_distance, remove_outliers=remove_outliers)
    if sp!=None:
        df_1 = sp.summary()
        df_2 = sp.summary(grp_chan=True, grp_stage=True)

        to_return = {}
        plt.figure("spindles_plot")
        sp.plot_average(center='Peak', time_before=1, time_after=1)
        # plt.show()
        # html_str = mpld3.fig_to_html(fig)
        # to_return["figure"] = html_str
        #  Temporarilly saved in root directory should change to commented
        # plt.savefig(NeurodesktopStorageLocation + '/spindles.png')
        plt.savefig(
            get_local_storage_path(workflow_id, run_id, step_id ) + "/output/" + 'spindles.png')
        # Transpose dataframes and add id column
        # df_1 = df_1.T
        df_1.insert(0, 'id', range(1, 1 + len(df_1)))

        # df_2 = df_2.T
        df_2.insert(0, 'id', range(1, 1 + len(df_2)))
        return {'data_frame_1': df_1.to_json(orient='records'), 'data_frame_2':df_2.to_json(orient='records')}
    else:
        return {'No spindles detected'}

@router.get("/sw_detect_two_dataframes")
async def sw_detect_two_dataframes(workflow_id: str,
                                   step_id: str,
                                   run_id: str,
                                   freq_sw: list[float] | None = Query(default=[0.3,1.5]),
                                   dur_neg: list[float] | None = Query(default=[0.3,1.5]),
                                   dur_pos: list[float] | None = Query(default=[0.1,1]),
                                   amp_neg: list[int] | None = Query(default=[40,200]),
                                   amp_pos: list[int] | None = Query(default=[10,150]),
                                   amp_ptp: list[int] | None = Query(default=[75,350]),
                                   include: list[int] | None = Query(default=[2,3]),
                                   remove_outliers: bool | None = Query(default=True),
                                   coupling: bool | None = Query(default=True),
                                   current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):
    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    data = mne.io.read_raw_fif(path_to_storage + "/" + files["edf"])
    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])

    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    sf = info['sfreq']
    hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)

    sw = yasa.sw_detect(raw_data, sf=sf, hypno=hypno,
                        coupling=coupling,remove_outliers=remove_outliers, include=include, freq_sw=freq_sw, dur_pos=dur_pos,
                        dur_neg=dur_neg, amp_neg=amp_neg, amp_pos=amp_pos, amp_ptp=amp_ptp)
    if sw!=None:
        df_1 = sw.summary()
        df_2 = sw.summary(grp_chan=True, grp_stage=True)

        to_return = {}
        # plt.figure("rose_plot")
        # ax = plt.subplot(projection='polar')
        figure_2 = rose_plot(workflow_id, run_id, step_id, df_1['PhaseAtSigmaPeak'], density=False, offset=0, lab_unit='degrees', start_zero=False)

        # plt.savefig(NeurodesktopStorageLocation + '/rose_plot.png')
        to_return['figure_2'] = figure_2


        plt.figure("slowwaves_plot")
        pg.plot_circmean(df_1['PhaseAtSigmaPeak'])
        print('Circular mean: %.3f rad' % pg.circ_mean(df_1['PhaseAtSigmaPeak']))
        print('Vector length: %.3f' % pg.circ_r(df_1['PhaseAtSigmaPeak']))
        # plt.show()
        # html_str = mpld3.fig_to_html(fig)
        # to_return["figure"] = html_str
        #  Temporarilly saved in root directory should change to commented
        # plt.savefig(NeurodesktopStorageLocation + '/slowwaves.png')
        plt.savefig(
            get_local_storage_path(workflow_id, run_id, step_id ) + "/output/" + 'slowwaves.png')
        # Transpose dataframes and add id column
        df_1_old = df_1
        # df_1 = df_1.T
        df_1.insert(0, 'id', range(1, 1 + len(df_1)))

        df_2_old = df_2
        # df_2 = df_2.T
        df_2.insert(0, 'id', range(1, 1 + len(df_2)))

        return {'data_frame_1':df_1.to_json(orient='records'), 'data_frame_2':df_2.to_json(orient='records'),
                'circular_mean:': pg.circ_mean(df_1_old['PhaseAtSigmaPeak']), # Circular mean (rad)
                'vector_length:': pg.circ_r(df_2_old['PhaseAtSigmaPeak'])} # Vector length (rad)
    else:
        return {'No slow-waves detected'}

@router.get("/PAC_values")
async def calculate_pac_values(workflow_id: str,
                               step_id: str,
                               run_id: str,
                               window: int | None = Query(default=15),
                               step: int | None = Query(default=15),
                               current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):


    to_return = {}
    plt.figure("pac_values_plot")

    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    data = mne.io.read_raw_fif(path_to_storage + "/" + files["edf"])
    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])


    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    sf = info['sfreq']
    hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)
    hypnoN2index = hypno == 2
    hypnoN3index = hypno == 3
    hypnoN2N3 = hypnoN2index + hypnoN3index

    # Segment N2 sleep into 15-seconds  non-overlapping epochs
    _, data_N2N3 = yasa.sliding_window(raw_data[0, hypnoN2N3], sf, window=window, step=step)

    # First, let's define our array of frequencies for phase and amplitude
    f_pha = np.arange(0.125, 4.25, 0.25)  # Frequency for phase
    f_amp = np.arange(7.25, 25.5, 0.5)  # Frequency for amplitude

    # Define a PAC object
    p = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp, verbose='WARNING')  # PAC method (2) equal to Modulation Index

    # Filter the data and extract the PAC values
    xpac = p.filterfit(sf, data_N2N3)

    sns.set(font_scale=1.1, style='white')

    # Plot the comodulogram
    p.comodulogram(xpac.mean(-1), title=str(p), vmin=0, plotas='contour', ncontours=100)
    plt.gca()
    # plt.show()

    # html_str = mpld3.fig_to_html(fig)
    # to_return["figure"] = html_str
    #  Temporarilly saved in root directory should change to commented
    # plt.savefig(NeurodesktopStorageLocation + '/pac_values.png')
    plt.savefig(
        get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'pac_values.png')
    return {'Figure':to_return}

@router.get("/extra_PAC_values")
async def calculate_extra_pac_values(workflow_id: str,
                                     step_id: str,
                                     run_id: str,
                                     window: int | None = Query(default=15),
                                     step: int | None = Query(default=15),
                                     current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):

    to_return = {}
    plt.figure("extra_pac_values_plot")

    files = get_files_for_slowwaves_spindle(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    data = mne.io.read_raw_fif(path_to_storage + "/" + files["edf"])
    hypno = pd.read_csv(path_to_storage + "/" + files["csv"])

    channels = data.ch_names
    raw_data = data.get_data(channels[0])
    info = data.info
    sf = info['sfreq']
    hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)
    hypnoN2index = hypno == 2
    hypnoN3index = hypno == 3
    hypnoN2N3 = hypnoN2index + hypnoN3index

    # Segment N2 sleep into 15-seconds  non-overlapping epochs
    _, data_N2N3 = yasa.sliding_window(raw_data[0, hypnoN2N3], sf, window=window, step=step)

    # First, let's define our array of frequencies for phase and amplitude
    f_pha = np.arange(0.125, 4.25, 0.25)  # Frequency for phase
    f_amp = np.arange(7.25, 25.5, 0.5)  # Frequency for amplitude

    # Define a PAC object
    p = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp, verbose='WARNING')  # PAC method (2) equal to Modulation Index

    # Filter the data and extract the PAC values
    xpac = p.filterfit(sf, data_N2N3)

    ################################################################################################
    ################################################################################################
    ################################################################################################

    raw_data = data.get_data(channels[1])
    info = data.info
    sf = info['sfreq']

    # Segment N2 sleep into 15-seconds  non-overlapping epochs
    _, data_N2N3 = yasa.sliding_window(raw_data[0, hypnoN2N3], sf, window=window, step=step)

    # First, let's define our array of frequencies for phase and amplitude
    f_pha = np.arange(0.125, 4.25, 0.25)  # Frequency for phase
    f_amp = np.arange(7.25, 25.5, 0.5)  # Frequency for amplitude

    # Define a PAC object
    p = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp, verbose='WARNING')  # PAC method (2) equal to Modulation Index

    # Filter the data and extract the PAC values
    ypac = p.filterfit(sf, data_N2N3)

    ###############################################

    # mne requires that the first is represented by the number of trials (n_epochs)
    # Therefore, we transpose the output PACs of both conditions
    pac_r1 = np.transpose(xpac, (2, 0, 1))
    pac_r2 = np.transpose(ypac, (2, 0, 1))

    n_perm = 1000  # number of permutations
    tail = 1  # only inspect the upper tail of the distribution
    # perform the correction
    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
        [pac_r1, pac_r2], n_permutations=n_perm, tail=tail)

    # create new stats image with only significant clusters
    t_obs_plot = np.nan * np.ones_like(t_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.001:
            t_obs_plot[c] = t_obs[c]
            t_obs[c] = np.nan

    title = 'Cluster-based corrected differences\nbetween central electrodes from both sessions'
    p.comodulogram(t_obs, cmap='gray', vmin=0, vmax=0.5, colorbar=True)
    p.comodulogram(t_obs_plot, cmap='viridis', vmin=0, vmax=0.5, title=title, colorbar=False)
    plt.gca().invert_yaxis()
    # plt.show()

    # html_str = mpld3.fig_to_html(fig)
    # to_return["figure"] = html_str

    #  Temporarilly saved in root directory should change to commented
    # plt.savefig(NeurodesktopStorageLocation + '/extra_pac_values.png')
    plt.savefig(
        get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'extra_pac_values.png')
    return {'Figure': to_return}


# Spindles detection
# Annotations_to_add have the folowing format which follows the format of adding it to the file with mne
# [ [starts], [durations], [names]  ]
@router.get("/save_annotation_to_file")
async def save_annotation_to_file(
                          workflow_id: str,
                          step_id: str,
                          run_id: str,
                          name: str,
                          annotations_to_add: str,
                          file_used: str | None = Query("original", regex="^(original)$|^(printed)$")):
    # Open file
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    list_all = []

    for i in range(len(channels)):
        if name == channels[i]:
            print("----")
            # print(raw_data[0][i].tolist())
            # print( raw_data[1].tolist())
            values = raw_data[i].tolist()
            time = raw_data[1].tolist()
            plt.plot(values, time, color='red', marker='o')
            plt.title('Channel Sleep/Spindle', fontsize=14)
            plt.xlabel('time', fontsize=14)
            plt.ylabel('value', fontsize=14)
            plt.grid(True)
            plt.show()
            # plt.savefig(get_local_storage_path(step_id, run_id) + "/" +'plot.png')

            # SW_list = []
            #
            # fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            # seaborn.lineplot(data=df_sync, x='Time', y='Amplitude', hue='Channel', palette="plasma", ci=95, ax=ax)
            # # ax.legend(frameon=False, loc='lower right')
            # ax.set_xlim(df_sync['Time'].min(), df_sync['Time'].max())
            # ax.set_title('Average SW')
            # ax.set_xlabel('Time (sec)')
            # ax.set_ylabel('Amplitude (uV)')
    # # Add the new annotations
    # raw_data = data.get_data()
    # # info = data.info
    # # channels = data.ch_names
    # # list_all = []
    # print("---fff---")
    # print(type(raw_data))
    # print(type(data))
    # new_annotations = mne.Annotations([31, 187, 317], [8, 8, 8],
    #                                   ["Movement", "Movement", "Movement"])
    # print("NOW ANNOTATING")
    # data.set_annotations(new_annotations)

    # print("NOW EXPORTING")
    # mne.export.export_raw(get_local_storage_path(step_id, run_id) + "/" + "test_file_edf.edf", data)
    # Open EDF BROWSER


    return {'Saved Annotations'}


# TODO remove current user form param and ge from file
@router.get("/mne/open/eeg", tags=["mne_open_eeg"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def mne_open_eeg(workflow_id: str,
                       step_id: str,
                       run_id: str,
                       selected_montage: str | None = "",
                       current_user: str | None = None) -> dict:
    # # Create a new jupyter notebook with the id of the run and step for recognition
    # create_notebook_mne_plot(input_run_id, input_step_id)

    # Initiate ssh connection with neurodesk container
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")
    channel = ssh.invoke_shell()

    # print("get_neurodesk_display_id()")
    # print(get_neurodesk_display_id())
    channel.send("cd /home/user/neurodesktop-storage\n")
    channel.send("sudo chmod 777 config\n")
    channel.send("cd /home/user/neurodesktop-storage/config\n")
    channel.send("sudo bash get_display.sh\n")

    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # Close previous isntances of code for the user
    # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!! THIS USER MUST CHANGE TO CURRENTLY USED USER
    channel.send("pkill -INT edfbrowser -u user\n")

    # Get file name to open with EDFBrowser
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    name_of_file = get_single_edf_file_from_local_temp_storage(workflow_id, run_id, step_id)
    file_full_path = path_to_storage + "/" + name_of_file

    # Give permissions in working folder
    channel.send("sudo chmod a+rw /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id +"/neurodesk_interim_storage\n")

    # Opening EDFBrowser
    channel.send("cd /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id +"/neurodesk_interim_storage\n")
    # print("/home/user/EDFbrowser/edfbrowser /home/user/'" + file_full_path + "'\n")
    if selected_montage != "":
        print("Montage selected path")
        print("/home/user/EDFbrowser/edfbrowser '/home/user" + file_full_path + "' /home/user" + NeurodesktopStorageLocation + "/montages/" + selected_montage + "\n")
        channel.send("/home/user/EDFbrowser/edfbrowser '/home/user" + file_full_path + "' /home/user" + NeurodesktopStorageLocation + "/montages/" + selected_montage + "\n")
    else:
        channel.send("/home/user/EDFbrowser/edfbrowser '/home/user" + file_full_path + "'\n")

    # OLD VISUAL STUDIO CODE CALL and terminate
    # channel.send("pkill -INT code -u user\n")
    # channel.send("/neurocommand/local/bin/mne-1_0_0.sh\n")
    # channel.send("nohup /usr/bin/code -n /home/user/neurodesktop-storage/created_1.ipynb --extensions-dir=/opt/vscode-extensions --disable-workspace-trust &\n")


@router.get("/mne/open/mne", tags=["mne_open_eeg"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def mne_open_mne(workflow_id: str, step_id: str, run_id: str, current_user: str | None = None) -> dict:
    # # Create a new jupyter notebook with the id of the run and step for recognition
    # create_notebook_mne_plot(input_run_id, input_step_id)

    # Initiate ssh connection with neurodesk container
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")
    channel = ssh.invoke_shell()

    # print("get_neurodesk_display_id()")
    # print(get_neurodesk_display_id())
    channel.send("cd /home/user/neurodesktop-storage\n")
    channel.send("sudo chmod 777 config\n")
    channel.send("cd /home/user/neurodesktop-storage/config\n")
    channel.send("sudo bash get_display.sh\n")

    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # Close previous isntances of code for the user
    # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!! THIS USER MUST CHANGE TO CURRENTLY USED USER
    channel.send("pkill -INT code -u user\n")
    channel.send("/neurocommand/local/bin/mne-1_0_0.sh\n")


    # Get file name to open with EDFBrowser
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    file_full_path = path_to_storage + "/" + name_of_file

    # Give permissions in working folder
    channel.send(
        "sudo chmod a+rw /home/user/neurodesktop-storage/runtime_config/workflow_" + workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage\n")

    channel.send("nohup /usr/bin/code -n /home/user/neurodesktop-storage/runtime_config/workflow_" +workflow_id + "/run_" + run_id + "/step_" + step_id + "/neurodesk_interim_storage/" + "created_1.ipynb --extensions-dir=/opt/vscode-extensions --disable-workspace-trust &\n")



# TODO chagne parameter name
@router.get("/return_signal", tags=["return_signal"])
# Start date time is returned as miliseconds epoch time
async def return_signal(workflow_id: str, step_id: str, run_id: str,input_name: str) -> dict:
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    raw_data = data.get_data(return_times=True)
    channels = data.ch_names

    for i in range(len(channels)):
        if input_name == channels[i]:

            to_return = {}
            to_return["signal"] = raw_data[0][i].tolist()
            to_return["signal_time"] = raw_data[1].tolist()
            to_return["start_date_time"] = data.info["meas_date"].timestamp() * 1000
            to_return["sfreq"] = data.info["sfreq"]

            # print(data.info["meas_date"].timestamp())
            # print(datetime.fromtimestamp(data.info["meas_date"].timestamp()))
            return to_return
    return {'Channel not found'}


@router.get("/mne/return_annotations", tags=["mne_return_annotations"])
async def mne_return_annotations(workflow_id: str, step_id: str, run_id: str, file_name: str | None = "annotation_test.csv") -> dict:
    # Default value probably isnt needed in final implementation
    annotations = get_annotations_from_csv(file_name)
    return annotations





@router.post("/receive_notebook_and_selection_configuration", tags=["receive__notebook_and_selection_configuration"])
async def receive_notebook_and_selection_configuration(input_config: ModelNotebookAndSelectionConfiguration,workflow_id: str, step_id: str, run_id: str,file_used: str | None = Query("original", regex="^(original)$|^(printed)$")) -> dict:
    # TODO TEMP
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)

    # data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)

    raw_data = data.get_data(return_times=True)

    print(input_config)
    # Produce new notebook
    create_notebook_mne_modular(
                                workflow_id=workflow_id,
                                run_id=run_id,
                                step_id=step_id,
                                file_to_save="created_1",
                                file_to_open="trial_av.edf",
                                notches_enabled=input_config.notches_enabled,
                                notches_length= input_config.notches_length,
                                annotations=True,
                                bipolar_references=input_config.bipolar_references,
                                reference_type= input_config.type_of_reference,
                                reference_channels_list=input_config.channels_reference,
                                selection_start_time= input_config.selection_start_time,
                                selection_end_time= input_config.selection_end_time,
                                repairing_artifacts_ica=input_config.repairing_artifacts_ica,
                                n_components=input_config.n_components,
                                list_exclude_ica=input_config.list_exclude_ica,
                                ica_method=input_config.ica_method
                                )

    # If there is a selection channel we need to crop
    if input_config.selection_channel != "":
        # data.crop(float(input_config.selection_start_time), float(input_config.selection_end_time))
        # data.save("/neurodesktop-storage/trial_av_processed.fif", "all", float(input_config.selection_start_time), float(input_config.selection_end_time), overwrite = True, buffer_size_sec=24)
        data.save(NeurodesktopStorageLocation + "/trial_av_processed.fif", "all", overwrite=True, buffer_size_sec=None)
    else:
        data.save(NeurodesktopStorageLocation + "/trial_av_processed.fif", "all", overwrite=True, buffer_size_sec=None)

    return {'Channel not found'}


@router.post("/receive_channel_selection", tags=["receive_channel_selection"])
async def receive_channel_selection(input_selection_channel: ModelSelectionChannelReference) -> dict:
    print(input_selection_channel)
    return {'Channel not found'}


# @router.get("/mne/return_annotations/watch", tags=["mne_return_annotations_watch"])
# async def mne_return_annotations_watch(file_name: str | None = "annotation_test.csv") -> dict:
#     # Default value proable isnt needed in final implementation
#
#     class MyHandler(FileSystemEventHandler):
#         def on_modified(self, event):
#             print(f'event type: {event.event_type}  path : {event.src_path}')
#             annotations = get_annotations_from_csv(file_name)
#             requests.post(url='http://localhost:3000/', data={'annotations': annotations})
#
#
#     event_handler = MyHandler()
#     observer = Observer()
#     observer.schedule(event_handler, path='/data/', recursive=False)
#     observer.start()
#
#     # try:
#     #     while True:
#     #         time.sleep(1)
#     # except KeyboardInterrupt:
#     #     observer.stop()
#     # observer.join()
#     return "Success"


@router.get("/mne/create_notebook", tags=["mne_create_notebook"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def mne_create_notebook(file_name: str,
                              notch_filter: int,
                              bipolar_reference: str,
                              average_reference: str,
                        ) -> dict:
    file_to_save = ""
    file_to_open = ""
    annotations = ""

    create_notebook_mne_modular(file_to_save,
                                file_to_open,
                                notch_filter,
                                annotations,
                                bipolar_reference,
                                average_reference)
    # create_notebook_mne_plot("hello", "again")

# TODO
# @router.get("/test/montage", tags=["test_montage"])
# async def test_montage() -> dict:
#     raw_data = data.get_data()
#     info = data.info
#     print('\nBUILT-IN MONTAGE FILES')
#     print('======================')
#     print(info)
#     print(raw_data)
#     ten_twenty_montage = mne.channels.make_standard_montage('example_data/trial_av')
#     print(ten_twenty_montage)

    # create_notebook_mne_plot("hello", "again")

@router.get("/get/montages", tags=["get_montages"])
async def get_montages() -> dict:
    """This function returns a list of the existing montages, which are files saved in neurodesktop_strorage montages"""
    print(NeurodesktopStorageLocation + "/montages")
    files_to_return = [f for f in os.listdir(NeurodesktopStorageLocation + '/montages') if isfile(join(NeurodesktopStorageLocation + '/montages', f))]
    return files_to_return


# @router.get("/test/notebook", tags=["test_notebook"])
# # Validation is done inline in the input of the function
# async def test_notebook(input_test_name: str, input_slices: str,
#                         ) -> dict:
#     create_notebook_mne_plot("hello", "again")


# @router.get("/test/mne", tags=["test_notebook"])
# # Validation is done inline in the input of the function
# async def test_mne() -> dict:
#     mne.export.export_raw(NeurodesktopStorageLocation + "/export_data_fixed.edf", data, physical_range=(-4999.84, 4999.84), overwrite=True)
#     mne.export.export_raw(NeurodesktopStorageLocation + "/export_data_not.edf", data, overwrite=True)
#

@router.get("/envelope_trend", tags=["envelope_trend"])
# Validation is done inline in the input of the function
async def return_envelopetrend(
                               workflow_id: str,
                               step_id: str,
                               run_id: str,
                               input_name: str,
                               window_size: int | None = None,
                               percent: float | None = None,
                               input_method: str | None = Query("none", regex="^(Simple)$|^(Cumulative)$|^(Exponential)$"),
                               file_used: str | None = Query("original", regex="^(original)$|^(printed)$")) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, workflow_id, run_id, step_id)
    raw_data = data.get_data()
    channels = data.ch_names

    for i in range(len(channels)):
        if input_name == channels[i]:
            if input_method == 'Simple':
                j = 1
                # Initialize an empty list to store cumulative moving averages
                moving_averages = []
                # Loop through the array elements
                while j < len(raw_data[i]) - window_size + 1:
                    # Calculate the average of current window
                    window_average = round(np.sum(raw_data[i][
                                                  j:j + window_size]) / window_size, 15)
                    window_average_upper = window_average * (1 + percent)
                    window_average_lower = window_average * (1 - percent)

                    # Store the average of current window in moving average list
                    row_to_append = {'date': (data.info["meas_date"].timestamp() * 1000 + data.times[j-1] * 1000).tolist(), 'signal': raw_data[i][j-1].tolist(), 'upper': window_average_upper, 'lower': window_average_lower}
                    moving_averages.append(row_to_append)

                    # Shift window to right by one position
                    j += 1
                return moving_averages
            elif input_method == 'Cumulative':
                j = 1
                # Initialize an empty list to store cumulative moving averages
                moving_averages = []
                # Store cumulative sums of array in cum_sum array
                cum_sum = np.cumsum(raw_data[i])
                # Loop through the array elements
                print(len(raw_data[i]))
                while j <= len(raw_data[i]):
                    # Calculate the cumulative average by dividing cumulative sum by number of elements till
                    # that position
                    window_average = round(cum_sum[j - 1] / j, 15)
                    window_average_upper = window_average * (1 + percent)
                    window_average_lower = window_average * (1 - percent)

                    row_to_append = {
                        'date': (data.info["meas_date"].timestamp() * 1000 + data.times[j-1] * 1000).tolist(),
                        'signal': raw_data[i][j-1].tolist(), 'upper': window_average_upper, 'lower': window_average_lower}
                    moving_averages.append(row_to_append)

                    # Shift window to right by one position
                    j += 1
                return moving_averages
            elif input_method == 'Exponential':
                x = 2 / (window_size + 1)  # smoothening factor
                j = 1
                # Initialize an empty list to store exponential moving averages
                moving_averages = []
                arr = []
                # Insert first exponential average in the list
                arr.append(raw_data[i][0])

                # Loop through the array elements
                while j < len(raw_data[i]):
                    # Calculate the exponential average by using the formula
                    window_average = round((x * raw_data[i][j]) +
                                           (1 - x) * arr[-1], 15)
                    window_average_upper = window_average * (1 + percent)
                    window_average_lower = window_average * (1 - percent)
                    arr.append(window_average)

                    row_to_append = {
                        'date': (data.info["meas_date"].timestamp() * 1000 + data.times[j-1] * 1000).tolist(),
                        'signal': raw_data[i][j-1].tolist(), 'upper': window_average_upper, 'lower': window_average_lower}
                    moving_averages.append(row_to_append)
                    # Shift window to right by one position
                    j += 1
                return moving_averages
            else:
                return {'Channel not found'}
    return {'Channel not found'}


@router.get("/back_average", tags=["back_average"])
async def back_average(
        workflow_id: str,
        step_id: str,
        run_id: str,
        input_name: str,
        # pick_channel: list | None = ["F4-Ref"],
        time_before_event: float | None = None,
        time_after_event: float | None = None,
        min_ptp_amplitude: float | None = None,
        max_ptp_amplitude: float | None = None,
        volt_display_minimum: float | None = None,
        volt_display_maximum: float | None = None,
        annotation_name: str | None = None,
):
    """This function applies a back average"""

    # Load the file from local or interim EDFBrowser storage
    data = load_file_from_local_or_interim_edfbrowser_storage("original", workflow_id, run_id, step_id)

    # Extract events from annotations
    events = mne.events_from_annotations(data, regexp=annotation_name)

    # Apply epochs without any baseline correction
    # This function uses only the channels that are marked as "data" in the montage
    # epochs = mne.Epochs(raw = data, picks= "data" ,events = events[0], tmin = time_before_event, tmax = time_after_event, reject= {'eeg':max_ptp_amplitude}, flat= {'eeg' :min_ptp_amplitude}, baseline=None)
    # Get list of channel names in edf of files of type eeg and emg
    # data = data.pick()
    print("TEST")
    raw_data = data.get_data()
    print(raw_data[1])

    # epochs = mne.Epochs(raw=data, picks=['emg', 'eeg'], events = events[0],tmin = time_before_event, tmax = time_after_event, baseline=None)
    epochs = mne.Epochs(raw=data, picks=['eeg'], events = events[0],tmin = time_before_event, tmax = time_after_event,  reject= {'eeg':max_ptp_amplitude}, flat= {'eeg' :min_ptp_amplitude}, reject_tmin=time_before_event, reject_tmax=time_after_event, baseline=None)

    # epochs = mne.Epochs(raw = data, picks="data",events = events[0],tmin = time_before_event, tmax = time_after_event, baseline=None)

    print(epochs)
    raw_data = epochs.get_data()
    print(raw_data)

    # epochs.plot(picks=["F4-Ref"], show=True)
    # evoked = epochs.average()

    # number_of_applicable_channels = len(epochs.ch_names)
    plt.figure()
    plt.rcParams["figure.figsize"] = (20, 200)
    # fig, axs = plt.subplots(number_of_applicable_channels, sharex=True, sharey=True)
    # plt.rcParams['figure.figsize'] = [80, 80]
    created_plots = []
    for channel in epochs.ch_names:
        evoked = epochs.average(picks=[channel], by_event_type=False)
        print(channel)
        print("volt_display_minimum")
        print(volt_display_minimum)
        fig_evoked = evoked.plot(titles=channel, ylim=dict(eeg=[volt_display_minimum, volt_display_maximum]))
        fig_evoked.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'temp_plot_'+channel+'.png')
        created_plots.append(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'temp_plot_'+channel+'.png')


    # epochs.plot_image
    evoked = epochs.average(picks="data", by_event_type=False)
    plot = evoked.plot(show=True)
    plot.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'back_average_plot.png')

    # evoked = epochs.average(picks="F4-Ref", by_event_type=False)
    # plot = evoked.plot(show=True)
    # plot.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'back_average_plot_f4.png', bbox_inches='tight')

    # for ax, channel in zip(axs,epochs.ch_names):
    #     evoked = epochs.average(picks=channel, by_event_type=False)
    #     evoked.plot(tit)
    print("Reached HERE")
    print(type(created_plots))
    print(created_plots)

    fig = plt.figure(figsize=(10,10))
    fig, axs = plt.subplots(len(created_plots),1)

    for ax, created_plot in zip(axs, created_plots):
        img = mpimg.imread(created_plot)
        # print("REACXHED HERE TOO")
        # print(img)
        ax.imshow(img)
        ax.axis('off')  # to hide axis

    plt.savefig(get_local_storage_path(workflow_id, step_id, run_id) + "/output/" + 'back_average_plot.png', bbox_inches='tight' )
    plt.show()
    # plot.savefig(NeurodesktopStorageLocation + '/back_average_plot.png')
    print(evoked)

    # mne.viz.plot_evoked(evoked, show=True)
    to_return = {}
    to_return["channels"] = epochs.ch_names
    return to_return

    return True

@router.get("/sleep_analysis_luiz")
async def sleepanalysislouiz(workflow_id: str,
                             step_id: str,
                             run_id: str):

    df_first_hypnos = []
    df_second_hypnos = []
    df_first_fif_files = []
    df_second_fif_files = []
    for entries in os.listdir('UU_Sleep'):
        if entries.endswith(".csv"):
            if entries.startswith("Subject A") or entries.startswith("Subject B"):
                df_first_hypnos.append(entries)
            else:
                df_second_hypnos.append(entries)
        elif entries.endswith(".fif"):
            if entries.startswith("Subject A") or entries.startswith("Subject B"):
                df_first_fif_files.append(entries)
            else:
                df_second_fif_files.append(entries)

    print(df_first_fif_files)
    print(df_first_hypnos)
    print(df_second_hypnos)
    print(df_second_fif_files)
    first_group_fif_files = []
    for entries in df_first_fif_files:
        path = 'UU_Sleep/' + entries
        data = mne.io.read_raw_fif(path)
        info = data.info
        raw_data = data.get_data()
        channels = data.ch_names

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        #first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        first_group_fif_files.append(df_signals)
        #print(df_signals)


    second_group_fif_files = []
    for entries in df_second_fif_files:
        path = 'UU_Sleep/' + entries
        data = mne.io.read_raw_fif(path)
        raw_data = data.get_data()
        channels = data.ch_names

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        # first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        second_group_fif_files.append(df_signals)
        # print(df_signals)

    info = mne.create_info(ch_names=channels, sfreq=info['sfreq'])
    array_list_first = []
    for i in range(len(first_group_fif_files)):
        array_list_first.append(first_group_fif_files[i].to_numpy())

    array_list_second = []
    for i in range(len(second_group_fif_files)):
        array_list_second.append(second_group_fif_files[i].to_numpy())

    first_mneraw_list = []
    second_mneraw_list = []
    for i in range(len(array_list_first)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_first[i]).T, info)
        first_mneraw_list.append(temp_MNEraw)

    for i in range(len(array_list_second)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_second[i]).T, info)
        second_mneraw_list.append(temp_MNEraw)

    ######## hypnogram


    #print(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))
    #yasa.plot_hypnogram(np.squeeze(df.to_numpy()))

    list_first_hypnos = []
    list_second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        list_first_hypnos.append(np.squeeze(df.to_numpy()))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        list_second_hypnos.append(np.squeeze(df.to_numpy()))

    sleep_stats_first = []
    sleep_stats_second = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_first.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_second.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    df_first_sleep_statistics = pd.DataFrame(sleep_stats_first)
    df_second_sleep_statistics = pd.DataFrame(sleep_stats_second)

    #sleep transition matrix

    counts_first = []
    probs_first = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_first.append(counts)
        probs_first.append(probs.round(3))

    counts_second = []
    probs_second = []
    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_second.append(counts)
        probs_second.append(probs.round(3))

    #Concatenate lists
    Sleepmatrix_counts_list = counts_first + counts_second
    Sleepmatrix_probs_list = probs_first + probs_second

    #Worthless transitions
    worthless_transitions = []
    worthless_first_transitions = []
    worthless_second_transitions = []

    for i in range(len(Sleepmatrix_counts_list)):
        temp = Sleepmatrix_counts_list[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_transitions))

    for i in range(len(counts_first)):
        temp = counts_first[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_first_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_first_transitions))

    for i in range(len(counts_second)):
        temp = counts_second[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_second_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_second_transitions))

    #probs
    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_first)):
        temp = probs_first[i][0:1]
        temp2 = probs_first[i][1:2]
        temp3 = probs_first[i][2:3]
        temp4 = probs_first[i][3:4]
        temp5 = probs_first[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans)
    df_N1_trans = pd.concat(N1_trans)
    df_N2_trans = pd.concat(N2_trans)
    df_N3_trans = pd.concat(N3_trans)
    df_REM_trans = pd.concat(REM_trans)

    first_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print(first_probs_Sleep_Matrix)

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_second)):
        temp = probs_second[i][0:1]
        temp2 = probs_second[i][1:2]
        temp3 = probs_second[i][2:3]
        temp4 = probs_second[i][3:4]
        temp5 = probs_second[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans)
    df_N1_trans = pd.concat(N1_trans)
    df_N2_trans = pd.concat(N2_trans)
    df_N3_trans = pd.concat(N3_trans)
    df_REM_trans = pd.concat(REM_trans)

    second_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print(second_probs_Sleep_Matrix)

    # Sleep fragmentation from probs

    sleep_stability_list = []
    for i in range(len(Sleepmatrix_probs_list)):
        stability_temp = np.diag(Sleepmatrix_probs_list[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list.append(stability_temp)

    df_sleep_stability_all = pd.DataFrame(sleep_stability_list)
    print(df_sleep_stability_all)

    sleep_stability_list_first = []
    for i in range(len(probs_first)):
        stability_temp = np.diag(probs_first[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_first.append(stability_temp)

    df_sleep_stability_first = pd.DataFrame(sleep_stability_list_first)
    print(df_sleep_stability_first)

    sleep_stability_list_second = []
    for i in range(len(probs_second)):
        stability_temp = np.diag(probs_second[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_second.append(stability_temp)

    df_sleep_stability_second = pd.DataFrame(sleep_stability_list_second)
    print(df_sleep_stability_second)

    #average
    z = 0
    for s in probs_first:
        z = z + s
    first_average_probs = z / len(probs_first)
    print(first_average_probs)

    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5, 5))
    sns.heatmap(first_average_probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax,
                cmap='YlGnBu', annot=True, fmt='.3f', cbar_kws={"orientation": "horizontal", "fraction":0.1,
                                                                "label":"Transition Probability"})

    ax.set_xlabel("To sleep stage")
    ax.xaxis.tick_top()
    ax.set_ylabel("From sleep stage")
    ax.xaxis.set_label_position('top')
    plt.rcParams["figure.dpi"] = 150
    plt.show()

    #Spectrograms and Bandpowers
    first_hypnos = []
    second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        first_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1/30, data=first_mneraw_list[i]))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        second_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1 / 30, data=second_mneraw_list[i]))

    #example
    data = first_mneraw_list[1].get_data()
    chan = first_mneraw_list[1].ch_names
    sf = first_mneraw_list[1].info['sfreq']
    fig = yasa.plot_spectrogram(data[chan.index("F4_A1_fil")], sf, first_hypnos[1])
    plt.show()

    #bandpower
    bandpower_first = []
    bandpower_second = []
    for i in range(len(df_first_hypnos)):
        bandpower_first.append(yasa.bandpower(first_mneraw_list[i], hypno=first_hypnos[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(bandpower_first)
    print(df_bandpower_first)

    for i in range(len(df_second_hypnos)):
        bandpower_second.append(yasa.bandpower(second_mneraw_list[i], hypno=second_hypnos[i], include=(2,3,4)))

    df_bandpower_second = pd.concat(bandpower_second)
    print(df_bandpower_second)



@router.get("/sleep_analysis_sensitivity_luiz")
async def sleepanalysislouizsensitivity(workflow_id: str,
                                        step_id: str,
                                        run_id: str):

    df_first_hypnos = []
    df_second_hypnos = []
    df_first_fif_files = []
    df_second_fif_files = []
    for entries in os.listdir('UU_Sleep'):
        if entries.endswith(".csv"):
            if entries.startswith("Subject A") or entries.startswith("Subject B"):
                df_first_hypnos.append(entries)
            else:
                df_second_hypnos.append(entries)
        elif entries.endswith(".fif"):
            if entries.startswith("Subject A") or entries.startswith("Subject B"):
                df_first_fif_files.append(entries)
            else:
                df_second_fif_files.append(entries)

    print(df_first_fif_files)
    print(df_first_hypnos)
    print(df_second_hypnos)
    print(df_second_fif_files)
    duration_first = []
    first_group_fif_files = []
    for entries in df_first_fif_files:
        path = 'UU_Sleep/' + entries
        data = mne.io.read_raw_fif(path)
        info = data.info
        raw_data = data.get_data()
        channels = data.ch_names
        duration_first.append(raw_data.shape[1]/info['sfreq'])

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        #first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        first_group_fif_files.append(df_signals)
        #print(df_signals)


    duration_second = []
    second_group_fif_files = []
    for entries in df_second_fif_files:
        path = 'UU_Sleep/' + entries
        data = mne.io.read_raw_fif(path)
        raw_data = data.get_data()
        channels = data.ch_names
        info = data.info
        duration_second.append(raw_data.shape[1] / info['sfreq'])

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        # first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        second_group_fif_files.append(df_signals)
        # print(df_signals)

    info = mne.create_info(ch_names=channels, sfreq=info['sfreq'])
    array_list_first = []
    for i in range(len(first_group_fif_files)):
        array_list_first.append(first_group_fif_files[i].to_numpy())

    array_list_second = []
    for i in range(len(second_group_fif_files)):
        array_list_second.append(second_group_fif_files[i].to_numpy())

    first_mneraw_list = []
    second_mneraw_list = []
    for i in range(len(array_list_first)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_first[i]).T, info)
        first_mneraw_list.append(temp_MNEraw)

    for i in range(len(array_list_second)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_second[i]).T, info)
        second_mneraw_list.append(temp_MNEraw)

    ######## hypnogram
    mne_raw_list = first_mneraw_list + second_mneraw_list
    firsthalf_mneraw_list = first_mneraw_list + second_mneraw_list
    secondhalf_mneraw_list = first_mneraw_list + second_mneraw_list


    #print(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))
    #yasa.plot_hypnogram(np.squeeze(df.to_numpy()))

    list_first_hypnos = []
    list_second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        list_first_hypnos.append(np.squeeze(df.to_numpy()))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        list_second_hypnos.append(np.squeeze(df.to_numpy()))

    hypno_list = list_first_hypnos + list_second_hypnos

    ##########################################################################
    ##########################################################################
    ##########################################################################
    ### Sensitivity Analysis 02 - trim all files to 06h window

    Hypno_sensitivity02_list_first = []
    sens02mnerawlist_first = []
    for i in range(len(list_first_hypnos)):
        temp = list_first_hypnos[i][:720]
        Hypno_sensitivity02_list_first.append(temp)
        sens02mnerawlist_first.append(first_mneraw_list[i])
        if duration_first[i] > 21600:
            sens02mnerawlist_first[i].crop(tmin=0, tmax=21600)

    Hypno_sensitivity02_list_second = []
    sens02mnerawlist_second = []
    for i in range(len(list_second_hypnos)):
        temp = list_second_hypnos[i][:720]
        Hypno_sensitivity02_list_second.append(temp)
        sens02mnerawlist_second.append(second_mneraw_list[i])
        if duration_second[i] > 21600:
            sens02mnerawlist_second[i].crop(tmin=0, tmax=21600)

    ##### Sensitivity Analysis 03

    hypno_first_half_list = []
    hypno_second_half_list = []
    sens03mnerawlist_first = []
    sens03mnerawlist_second = []
    for i in range(len(mne_raw_list)):
        x = hypno_list[i].size/2
        temp_first = hypno_list[i][:int(x)]
        temp_second = hypno_list[i][int(x):]
        hypno_first_half_list.append(temp_first)
        hypno_second_half_list.append(temp_second)
        sens03mnerawlist_first.append(firsthalf_mneraw_list[i].copy().crop(tmin=0, tmax=int(x*30)))
        sens03mnerawlist_second.append(secondhalf_mneraw_list[i].copy().crop(tmin=int(x*30)))

    ### Sensitivity Analysis 02 - hypnograms

    hypnosensitivity02list_all = Hypno_sensitivity02_list_first + Hypno_sensitivity02_list_second
    sensitivity02_sleepstatistics = []
    for i in range(len(hypnosensitivity02list_all)):
        sensitivity02_sleepstatistics.append(yasa.sleep_statistics(hypnosensitivity02list_all[i], sf_hyp=1/30))

    df_sens02sleep_statistics = pd.DataFrame(sensitivity02_sleepstatistics)
    print('Sensitivity 02 - Sleep Statistics')
    print(df_sens02sleep_statistics)

    # Sensitivity Analysis 03 - Hypnograms

    sleepstats_firsthalf_list = []
    sleepstats_secondhalf_list = []
    for i in range(len(hypno_first_half_list)):
        temp_df1 = yasa.sleep_statistics(hypno_first_half_list[i], sf_hyp=1/30)
        temp_df2 = yasa.sleep_statistics(hypno_second_half_list[i], sf_hyp=1/30)

        sleepstats_firsthalf_list.append(temp_df1)
        sleepstats_secondhalf_list.append(temp_df2)

    df_sens03sleep_statisticsfirsthalf = pd.DataFrame(sleepstats_firsthalf_list)
    print('Sensitivity 03 - Sleep Statistics - first half list')
    print(df_sens03sleep_statisticsfirsthalf)

    df_sens03sleep_statisticssecondhalf = pd.DataFrame(sleepstats_secondhalf_list)
    print('Sensitivity 03 - Sleep Statistics - second half list')
    print(df_sens03sleep_statisticssecondhalf)

    ######################################################################################
    sleep_stats_first = []
    sleep_stats_second = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_first.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_second.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    df_first_sleep_statistics = pd.DataFrame(sleep_stats_first)
    df_second_sleep_statistics = pd.DataFrame(sleep_stats_second)

    ########################################################################
    #sleep transition matrix

    ## Sensitivity Analysis 02

    sensitivity02_counts_list = []
    sensitivity02_probs_list = []
    for i in range(len(hypnosensitivity02list_all)):
        counts, probs = yasa.transition_matrix(hypnosensitivity02list_all[i])
        sensitivity02_counts_list.append(counts)
        sensitivity02_probs_list.append(probs.round(3))

    # Sensitivity Analysis 03

    sensitivity03_counts_list_firsthalf = []
    sensitivity03_probs_list_firsthalf = []

    sensitivity03_counts_list_secondhalf = []
    sensitivity03_probs_list_secondhalf = []
    for i in range(len(hypno_first_half_list)):
        counts, probs = yasa.transition_matrix(hypno_first_half_list[i])
        sensitivity03_counts_list_firsthalf.append(counts)
        sensitivity03_probs_list_firsthalf.append(probs.round(3))

        counts, probs = yasa.transition_matrix(hypno_second_half_list[i])
        sensitivity03_counts_list_secondhalf.append(counts)
        sensitivity03_probs_list_secondhalf.append(probs.round(3))

    ################################################

    counts_first = []
    probs_first = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_first.append(counts)
        probs_first.append(probs.round(3))

    counts_second = []
    probs_second = []
    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_second.append(counts)
        probs_second.append(probs.round(3))

    #Concatenate lists
    Sleepmatrix_counts_list = counts_first + counts_second
    Sleepmatrix_probs_list = probs_first + probs_second

    #Worthless transitions
    worthless_transitions = []
    worthless_first_transitions = []
    worthless_second_transitions = []
    worthless_firsthalf_transitions = []
    worthless_secondhalf_transitions = []
    worthless_sens02_transitions = []

    for i in range(len(Sleepmatrix_counts_list)):
        temp = Sleepmatrix_counts_list[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_transitions.append(temp)

        temp = sensitivity02_counts_list[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_sens02_transitions.append(temp)

        temp = sensitivity03_counts_list_firsthalf[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_firsthalf_transitions.append(temp)

        temp = sensitivity03_counts_list_secondhalf[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_secondhalf_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_transitions))

    print('Sensitivity 02')
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_sens02_transitions))

    print("Sensitivity 03 - first half")
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_firsthalf_transitions))

    print("Sensitivity 03 - second half")
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_secondhalf_transitions))

    for i in range(len(counts_first)):
        temp = counts_first[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_first_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_first_transitions))

    for i in range(len(counts_second)):
        temp = counts_second[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_second_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_second_transitions))

    #probs
    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_first)):
        temp = probs_first[i][0:1]
        temp2 = probs_first[i][1:2]
        temp3 = probs_first[i][2:3]
        temp4 = probs_first[i][3:4]
        temp5 = probs_first[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans)
    df_N1_trans = pd.concat(N1_trans)
    df_N2_trans = pd.concat(N2_trans)
    df_N3_trans = pd.concat(N3_trans)
    df_REM_trans = pd.concat(REM_trans)

    first_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print(first_probs_Sleep_Matrix)

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_second)):
        temp = probs_second[i][0:1]
        temp2 = probs_second[i][1:2]
        temp3 = probs_second[i][2:3]
        temp4 = probs_second[i][3:4]
        temp5 = probs_second[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans)
    df_N1_trans = pd.concat(N1_trans)
    df_N2_trans = pd.concat(N2_trans)
    df_N3_trans = pd.concat(N3_trans)
    df_REM_trans = pd.concat(REM_trans)

    second_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print(second_probs_Sleep_Matrix)

    # Sensitivity Analysis 02

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity02_probs_list)):
        temp = sensitivity02_probs_list[i][0:1]
        temp2 = sensitivity02_probs_list[i][1:2]
        temp3 = sensitivity02_probs_list[i][2:3]
        temp4 = sensitivity02_probs_list[i][3:4]
        temp5 = sensitivity02_probs_list[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans)
    df_N1_trans = pd.concat(N1_trans)
    df_N2_trans = pd.concat(N2_trans)
    df_N3_trans = pd.concat(N3_trans)
    df_REM_trans = pd.concat(REM_trans)

    sensitivity02_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print(sensitivity02_probs_Sleep_Matrix)

    # Sensitivity Analysis 03 - first half

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity03_probs_list_firsthalf)):
        temp = sensitivity03_probs_list_firsthalf[i][0:1]
        temp2 = sensitivity03_probs_list_firsthalf[i][1:2]
        temp3 = sensitivity03_probs_list_firsthalf[i][2:3]
        temp4 = sensitivity03_probs_list_firsthalf[i][3:4]
        temp5 = sensitivity03_probs_list_firsthalf[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans)
    df_N1_trans = pd.concat(N1_trans)
    df_N2_trans = pd.concat(N2_trans)
    df_N3_trans = pd.concat(N3_trans)
    df_REM_trans = pd.concat(REM_trans)

    sensitivity03_first_half_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print(sensitivity03_first_half_probs_Sleep_Matrix)

    # Sensitivity Analysis 03 - second half

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity03_probs_list_secondhalf)):
        temp = sensitivity03_probs_list_secondhalf[i][0:1]
        temp2 = sensitivity03_probs_list_secondhalf[i][1:2]
        temp3 = sensitivity03_probs_list_secondhalf[i][2:3]
        temp4 = sensitivity03_probs_list_secondhalf[i][3:4]
        temp5 = sensitivity03_probs_list_secondhalf[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans)
    df_N1_trans = pd.concat(N1_trans)
    df_N2_trans = pd.concat(N2_trans)
    df_N3_trans = pd.concat(N3_trans)
    df_REM_trans = pd.concat(REM_trans)

    sensitivity03_second_half_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print(sensitivity03_second_half_probs_Sleep_Matrix)

    # Sleep fragmentation from probs

    sleep_stability_list = []
    for i in range(len(Sleepmatrix_probs_list)):
        stability_temp = np.diag(Sleepmatrix_probs_list[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list.append(stability_temp)

    df_sleep_stability_all = pd.DataFrame(sleep_stability_list)
    print(df_sleep_stability_all)

    sleep_stability_list_first = []
    for i in range(len(probs_first)):
        stability_temp = np.diag(probs_first[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_first.append(stability_temp)

    df_sleep_stability_first = pd.DataFrame(sleep_stability_list_first)
    print(df_sleep_stability_first)

    sleep_stability_list_second = []
    for i in range(len(probs_second)):
        stability_temp = np.diag(probs_second[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_second.append(stability_temp)

    df_sleep_stability_second = pd.DataFrame(sleep_stability_list_second)
    print(df_sleep_stability_second)

    # Sensitivity 02

    sleep_stability_list_sens02 = []
    for i in range(len(sensitivity02_probs_list)):
        stability_temp = np.diag(sensitivity02_probs_list[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens02.append(stability_temp)

    df_sleep_stability_sens02 = pd.DataFrame(sleep_stability_list_sens02)
    print('Sensitivity 02')
    print(df_sleep_stability_sens02)

    # Sensitivity 03 - first half

    sleep_stability_list_sens03_firsthalf = []
    for i in range(len(sensitivity03_probs_list_firsthalf)):
        stability_temp = np.diag(sensitivity03_probs_list_firsthalf[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens03_firsthalf.append(stability_temp)

    df_sleep_stability_all_sens03_firsthalf = pd.DataFrame(sleep_stability_list_sens03_firsthalf)
    print(df_sleep_stability_all_sens03_firsthalf)

    # Sensitivity 03 - second half

    sleep_stability_list_sens03_secondhalf = []
    for i in range(len(sensitivity03_probs_list_secondhalf)):
        stability_temp = np.diag(sensitivity03_probs_list_secondhalf[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens03_secondhalf.append(stability_temp)

    df_sleep_stability_all_sens03_secondhalf = pd.DataFrame(sleep_stability_list_sens03_secondhalf)
    print(df_sleep_stability_all_sens03_secondhalf)

    #average
    z = 0
    for s in probs_first:
        z = z + s
    first_average_probs = z / len(probs_first)
    print(first_average_probs)

    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5, 5))
    sns.heatmap(first_average_probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax,
                cmap='YlGnBu', annot=True, fmt='.3f', cbar_kws={"orientation": "horizontal", "fraction":0.1,
                                                                "label":"Transition Probability"})

    ax.set_xlabel("To sleep stage")
    ax.xaxis.tick_top()
    ax.set_ylabel("From sleep stage")
    ax.xaxis.set_label_position('top')
    plt.rcParams["figure.dpi"] = 150
    plt.show()

    #Spectrograms and Bandpowers
    first_hypnos = []
    second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        first_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1/30, data=first_mneraw_list[i]))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        second_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1 / 30, data=second_mneraw_list[i]))

    #sensitivity_analysis_3

    hypnolist = []
    firsthalf_hypnolist = []
    secondhalf_hypnolist = []
    for i in range(len(hypno_first_half_list)):
        HYPNOup_temp = yasa.hypno_upsample_to_data(hypno_list[i], sf_hypno=1/30, data=mne_raw_list[i])
        HYPNOup_temp4 = yasa.hypno_upsample_to_data(hypno_first_half_list[i], sf_hypno=1/30, data=sens03mnerawlist_first[i])
        HYPNOup_temp5 = yasa.hypno_upsample_to_data(hypno_second_half_list[i], sf_hypno=1/30, data=sens03mnerawlist_second[i])

        hypnolist.append(HYPNOup_temp)
        firsthalf_hypnolist.append(HYPNOup_temp4)
        secondhalf_hypnolist.append(HYPNOup_temp5)


    #example
    data = first_mneraw_list[1].get_data()
    chan = first_mneraw_list[1].ch_names
    sf = first_mneraw_list[1].info['sfreq']
    fig = yasa.plot_spectrogram(data[chan.index("F4_A1_fil")], sf, first_hypnos[1])
    plt.show()

    #bandpower
    bandpower_stages_list = []
    bandpower_second = []

    for i in range(len(mne_raw_list)):
        bandpower_stages_list.append(yasa.bandpower(mne_raw_list[i], hypno=hypnolist[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(bandpower_stages_list)
    print(df_bandpower_first)

    #Sensitivity Analysis 03
    firsthalf_bandpowerlist = []
    secondhalf_bandpowerlist = []
    for i in range(len(firsthalf_hypnolist)):
        firsthalf_bandpowerlist.append(yasa.bandpower(sens03mnerawlist_first[i], hypno=firsthalf_hypnolist[i], include=(2,3,4)))
        secondhalf_bandpowerlist.append(yasa.bandpower(sens03mnerawlist_second[i], hypno=secondhalf_hypnolist[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(firsthalf_bandpowerlist)
    print("Sensitivity Analysis 03 - First Half")
    print(df_bandpower_first)

    df_bandpower_first = pd.concat(secondhalf_bandpowerlist)
    print("Sensitivity Analysis 03 - Second Half")
    print(df_bandpower_first)



@router.get("/sleep_analysis_sensitivity_luizaddsubject")
async def sleepanalysislouizsensitivityaddsubject(workflow_id: str,
                                                  step_id: str,
                                                  run_id: str):

    df_first_hypnos = []
    df_second_hypnos = []
    df_first_fif_files = []
    df_second_fif_files = []
    for entries in os.listdir('UU_Sleep'):
        if entries.endswith(".csv"):
            if entries.startswith("Subject A") or entries.startswith("Subject B"):
                df_first_hypnos.append(entries)
            else:
                df_second_hypnos.append(entries)
        elif entries.endswith(".fif"):
            if entries.startswith("Subject A") or entries.startswith("Subject B"):
                df_first_fif_files.append(entries)
            else:
                df_second_fif_files.append(entries)

    print(df_first_fif_files)
    print(df_first_hypnos)
    print(df_second_hypnos)
    print(df_second_fif_files)
    fif_files = df_first_fif_files + df_second_fif_files
    fif_files_subjects = []
    for i in range(len(fif_files)):
        fif_files_subjects.append(fif_files[i].split(".")[0])

    duration_first = []
    first_group_fif_files = []
    for entries in df_first_fif_files:
        path = 'UU_Sleep/' + entries
        data = mne.io.read_raw_fif(path)
        info = data.info
        raw_data = data.get_data()
        channels = data.ch_names
        duration_first.append(raw_data.shape[1]/info['sfreq'])

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        #first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        first_group_fif_files.append(df_signals)
        #print(df_signals)


    duration_second = []
    second_group_fif_files = []
    for entries in df_second_fif_files:
        path = 'UU_Sleep/' + entries
        data = mne.io.read_raw_fif(path)
        raw_data = data.get_data()
        channels = data.ch_names
        info = data.info
        duration_second.append(raw_data.shape[1] / info['sfreq'])

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        # first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        second_group_fif_files.append(df_signals)
        # print(df_signals)

    info = mne.create_info(ch_names=channels, sfreq=info['sfreq'])
    array_list_first = []
    for i in range(len(first_group_fif_files)):
        array_list_first.append(first_group_fif_files[i].to_numpy())

    array_list_second = []
    for i in range(len(second_group_fif_files)):
        array_list_second.append(second_group_fif_files[i].to_numpy())

    first_mneraw_list = []
    second_mneraw_list = []
    for i in range(len(array_list_first)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_first[i]).T, info)
        first_mneraw_list.append(temp_MNEraw)

    for i in range(len(array_list_second)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_second[i]).T, info)
        second_mneraw_list.append(temp_MNEraw)

    ######## hypnogram
    mne_raw_list = first_mneraw_list + second_mneraw_list
    firsthalf_mneraw_list = first_mneraw_list + second_mneraw_list
    secondhalf_mneraw_list = first_mneraw_list + second_mneraw_list


    #print(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))
    #yasa.plot_hypnogram(np.squeeze(df.to_numpy()))

    list_first_hypnos = []
    list_second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        list_first_hypnos.append(np.squeeze(df.to_numpy()))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        list_second_hypnos.append(np.squeeze(df.to_numpy()))

    hypno_list = list_first_hypnos + list_second_hypnos

    ##########################################################################
    ##########################################################################
    ##########################################################################
    ### Sensitivity Analysis 02 - trim all files to 06h window

    Hypno_sensitivity02_list_first = []
    sens02mnerawlist_first = []
    for i in range(len(list_first_hypnos)):
        temp = list_first_hypnos[i][:720]
        Hypno_sensitivity02_list_first.append(temp)
        sens02mnerawlist_first.append(first_mneraw_list[i])
        if duration_first[i] > 21600:
            sens02mnerawlist_first[i].crop(tmin=0, tmax=21600)

    Hypno_sensitivity02_list_second = []
    sens02mnerawlist_second = []
    for i in range(len(list_second_hypnos)):
        temp = list_second_hypnos[i][:720]
        Hypno_sensitivity02_list_second.append(temp)
        sens02mnerawlist_second.append(second_mneraw_list[i])
        if duration_second[i] > 21600:
            sens02mnerawlist_second[i].crop(tmin=0, tmax=21600)

    sens_02_mnerawlist_all = sens02mnerawlist_first + sens02mnerawlist_second
    hypnosensitivity02_all = Hypno_sensitivity02_list_first + Hypno_sensitivity02_list_second

    ##### Sensitivity Analysis 03

    hypno_first_half_list = []
    hypno_second_half_list = []
    sens03mnerawlist_first = []
    sens03mnerawlist_second = []
    for i in range(len(mne_raw_list)):
        x = hypno_list[i].size/2
        temp_first = hypno_list[i][:int(x)]
        temp_second = hypno_list[i][int(x):]
        hypno_first_half_list.append(temp_first)
        hypno_second_half_list.append(temp_second)
        sens03mnerawlist_first.append(firsthalf_mneraw_list[i].copy().crop(tmin=0, tmax=int(x*30)))
        sens03mnerawlist_second.append(secondhalf_mneraw_list[i].copy().crop(tmin=int(x*30)))

    ### Sensitivity Analysis 02 - hypnograms

    hypnosensitivity02list_all = Hypno_sensitivity02_list_first + Hypno_sensitivity02_list_second
    sensitivity02_sleepstatistics = []
    for i in range(len(hypnosensitivity02list_all)):
        sensitivity02_sleepstatistics.append(yasa.sleep_statistics(hypnosensitivity02list_all[i], sf_hyp=1/30))

    df_sens02sleep_statistics = pd.DataFrame(sensitivity02_sleepstatistics)
    df_fif_files_subjects = pd.DataFrame(fif_files_subjects, columns=['subjects'])
    df_sens02sleep_statistics = pd.concat([df_fif_files_subjects, df_sens02sleep_statistics], axis=1)
    print('Sensitivity 02 - Sleep Statistics')
    print(df_sens02sleep_statistics)

    # Sensitivity Analysis 03 - Hypnograms

    sleepstats_firsthalf_list = []
    sleepstats_secondhalf_list = []
    for i in range(len(hypno_first_half_list)):
        temp_df1 = yasa.sleep_statistics(hypno_first_half_list[i], sf_hyp=1/30)
        temp_df2 = yasa.sleep_statistics(hypno_second_half_list[i], sf_hyp=1/30)

        sleepstats_firsthalf_list.append(temp_df1)
        sleepstats_secondhalf_list.append(temp_df2)

    df_sens03sleep_statisticsfirsthalf = pd.DataFrame(sleepstats_firsthalf_list)
    df_sens03sleep_statisticsfirsthalf = pd.concat([df_fif_files_subjects, df_sens03sleep_statisticsfirsthalf], axis=1)
    print('Sensitivity 03 - Sleep Statistics - first half list')
    print(df_sens03sleep_statisticsfirsthalf)

    df_sens03sleep_statisticssecondhalf = pd.DataFrame(sleepstats_secondhalf_list)
    df_sens03sleep_statisticssecondhalf = pd.concat([df_fif_files_subjects, df_sens03sleep_statisticssecondhalf], axis=1)
    print('Sensitivity 03 - Sleep Statistics - second half list')
    print(df_sens03sleep_statisticssecondhalf)

    ######################################################################################
    sleep_stats_first = []
    sleep_stats_second = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_first.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_second.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    df_first_sleep_statistics = pd.DataFrame(sleep_stats_first)
    df_second_sleep_statistics = pd.DataFrame(sleep_stats_second)

    first_fif_files_subjects = []
    for i in range(len(df_first_fif_files)):
        first_fif_files_subjects.append(df_first_fif_files[i].split(".")[0])

    df_first_fif_files_dataframe = pd.DataFrame(first_fif_files_subjects, columns = ['subjects'])

    second_fif_files_subjects = []
    for i in range(len(df_second_fif_files)):
        second_fif_files_subjects.append(df_second_fif_files[i].split(".")[0])

    df_second_fif_files_dataframe = pd.DataFrame(second_fif_files_subjects, columns=['subjects'])

    df_first_sleep_statistics = pd.concat([df_first_fif_files_dataframe, df_first_sleep_statistics], axis = 1)
    print('First folder - Sleep Statistics')
    print(df_first_sleep_statistics)

    df_second_sleep_statistics = pd.concat([df_second_fif_files_dataframe, df_second_sleep_statistics], axis = 1)
    print('Second folder - Sleep Statistics')
    print(df_second_sleep_statistics)

    ########################################################################
    #sleep transition matrix

    ## Sensitivity Analysis 02

    sensitivity02_counts_list = []
    sensitivity02_probs_list = []
    for i in range(len(hypnosensitivity02list_all)):
        counts, probs = yasa.transition_matrix(hypnosensitivity02list_all[i])
        sensitivity02_counts_list.append(counts)
        sensitivity02_probs_list.append(probs.round(3))

    # Sensitivity Analysis 03

    sensitivity03_counts_list_firsthalf = []
    sensitivity03_probs_list_firsthalf = []

    sensitivity03_counts_list_secondhalf = []
    sensitivity03_probs_list_secondhalf = []
    for i in range(len(hypno_first_half_list)):
        counts, probs = yasa.transition_matrix(hypno_first_half_list[i])
        sensitivity03_counts_list_firsthalf.append(counts)
        sensitivity03_probs_list_firsthalf.append(probs.round(3))

        counts, probs = yasa.transition_matrix(hypno_second_half_list[i])
        sensitivity03_counts_list_secondhalf.append(counts)
        sensitivity03_probs_list_secondhalf.append(probs.round(3))

    ################################################

    counts_first = []
    probs_first = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_first.append(counts)
        probs_first.append(probs.round(3))

    counts_second = []
    probs_second = []
    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_second.append(counts)
        probs_second.append(probs.round(3))

    #Concatenate lists
    Sleepmatrix_counts_list = counts_first + counts_second
    Sleepmatrix_probs_list = probs_first + probs_second

    #Worthless transitions
    worthless_transitions = []
    worthless_first_transitions = []
    worthless_second_transitions = []
    worthless_firsthalf_transitions = []
    worthless_secondhalf_transitions = []
    worthless_sens02_transitions = []

    for i in range(len(Sleepmatrix_counts_list)):
        temp = Sleepmatrix_counts_list[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_transitions.append(temp)

        temp = sensitivity02_counts_list[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_sens02_transitions.append(temp)

        temp = sensitivity03_counts_list_firsthalf[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_firsthalf_transitions.append(temp)

        temp = sensitivity03_counts_list_secondhalf[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_secondhalf_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_transitions))

    print('Sensitivity 02')
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_sens02_transitions))

    print("Sensitivity 03 - first half")
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_firsthalf_transitions))

    print("Sensitivity 03 - second half")
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_secondhalf_transitions))

    for i in range(len(counts_first)):
        temp = counts_first[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_first_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_first_transitions))

    for i in range(len(counts_second)):
        temp = counts_second[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_second_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_second_transitions))

    #probs
    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_first)):
        temp = probs_first[i][0:1]
        temp2 = probs_first[i][1:2]
        temp3 = probs_first[i][2:3]
        temp4 = probs_first[i][3:4]
        temp5 = probs_first[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=df_first_fif_files)
    df_N1_trans = pd.concat(N1_trans, keys=df_first_fif_files)
    df_N2_trans = pd.concat(N2_trans, keys=df_first_fif_files)
    df_N3_trans = pd.concat(N3_trans, keys=df_first_fif_files)
    df_REM_trans = pd.concat(REM_trans, keys=df_first_fif_files)

    first_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("first folder")
    print(first_probs_Sleep_Matrix)

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_second)):
        temp = probs_second[i][0:1]
        temp2 = probs_second[i][1:2]
        temp3 = probs_second[i][2:3]
        temp4 = probs_second[i][3:4]
        temp5 = probs_second[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=df_second_fif_files)
    df_N1_trans = pd.concat(N1_trans, keys=df_second_fif_files)
    df_N2_trans = pd.concat(N2_trans, keys=df_second_fif_files)
    df_N3_trans = pd.concat(N3_trans, keys=df_second_fif_files)
    df_REM_trans = pd.concat(REM_trans, keys=df_second_fif_files)

    second_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("second folder")
    print(second_probs_Sleep_Matrix)

    # Sensitivity Analysis 02

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity02_probs_list)):
        temp = sensitivity02_probs_list[i][0:1]
        temp2 = sensitivity02_probs_list[i][1:2]
        temp3 = sensitivity02_probs_list[i][2:3]
        temp4 = sensitivity02_probs_list[i][3:4]
        temp5 = sensitivity02_probs_list[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=fif_files_subjects)
    df_N1_trans = pd.concat(N1_trans, keys=fif_files_subjects)
    df_N2_trans = pd.concat(N2_trans, keys=fif_files_subjects)
    df_N3_trans = pd.concat(N3_trans, keys=fif_files_subjects)
    df_REM_trans = pd.concat(REM_trans, keys=fif_files_subjects)

    sensitivity02_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("Sensitivity 02")
    print(sensitivity02_probs_Sleep_Matrix)

    # Sensitivity Analysis 03 - first half

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity03_probs_list_firsthalf)):
        temp = sensitivity03_probs_list_firsthalf[i][0:1]
        temp2 = sensitivity03_probs_list_firsthalf[i][1:2]
        temp3 = sensitivity03_probs_list_firsthalf[i][2:3]
        temp4 = sensitivity03_probs_list_firsthalf[i][3:4]
        temp5 = sensitivity03_probs_list_firsthalf[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=fif_files_subjects)
    df_N1_trans = pd.concat(N1_trans, keys=fif_files_subjects)
    df_N2_trans = pd.concat(N2_trans, keys=fif_files_subjects)
    df_N3_trans = pd.concat(N3_trans, keys=fif_files_subjects)
    df_REM_trans = pd.concat(REM_trans, keys=fif_files_subjects)

    sensitivity03_first_half_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("Sesnitivity Analysis - First half")
    print(sensitivity03_first_half_probs_Sleep_Matrix)

    # Sensitivity Analysis 03 - second half

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity03_probs_list_secondhalf)):
        temp = sensitivity03_probs_list_secondhalf[i][0:1]
        temp2 = sensitivity03_probs_list_secondhalf[i][1:2]
        temp3 = sensitivity03_probs_list_secondhalf[i][2:3]
        temp4 = sensitivity03_probs_list_secondhalf[i][3:4]
        temp5 = sensitivity03_probs_list_secondhalf[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=fif_files_subjects)
    df_N1_trans = pd.concat(N1_trans, keys=fif_files_subjects)
    df_N2_trans = pd.concat(N2_trans, keys=fif_files_subjects)
    df_N3_trans = pd.concat(N3_trans, keys=fif_files_subjects)
    df_REM_trans = pd.concat(REM_trans, keys=fif_files_subjects)

    sensitivity03_second_half_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("Sensitivity Analysis 03 - Second Half")
    print(sensitivity03_second_half_probs_Sleep_Matrix)

    # Sleep fragmentation from probs

    sleep_stability_list = []
    for i in range(len(Sleepmatrix_probs_list)):
        stability_temp = np.diag(Sleepmatrix_probs_list[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list.append(stability_temp)

    df_sleep_stability_all = pd.DataFrame(sleep_stability_list)
    df_sleep_stability_all = pd.concat([df_fif_files_subjects, df_sleep_stability_all], axis=1)
    print('Both folders')
    print(df_sleep_stability_all)

    sleep_stability_list_first = []
    for i in range(len(probs_first)):
        stability_temp = np.diag(probs_first[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_first.append(stability_temp)

    df_sleep_stability_first = pd.DataFrame(sleep_stability_list_first)
    df_sleep_stability_first = pd.concat([df_first_fif_files_dataframe, df_sleep_stability_first], axis=1)
    print("First Folder")
    print(df_sleep_stability_first)

    sleep_stability_list_second = []
    for i in range(len(probs_second)):
        stability_temp = np.diag(probs_second[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_second.append(stability_temp)

    df_sleep_stability_second = pd.DataFrame(sleep_stability_list_second)
    df_sleep_stability_second = pd.concat([df_second_fif_files_dataframe, df_sleep_stability_second], axis=1)
    print("Second Folder")
    print(df_sleep_stability_second)

    # Sensitivity 02

    sleep_stability_list_sens02 = []
    for i in range(len(sensitivity02_probs_list)):
        stability_temp = np.diag(sensitivity02_probs_list[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens02.append(stability_temp)

    df_sleep_stability_sens02 = pd.DataFrame(sleep_stability_list_sens02)
    df_sleep_stability_sens02 = pd.concat([df_fif_files_subjects, df_sleep_stability_sens02], axis=1)
    print('Sensitivity 02')
    print(df_sleep_stability_sens02)

    # Sensitivity 03 - first half

    sleep_stability_list_sens03_firsthalf = []
    for i in range(len(sensitivity03_probs_list_firsthalf)):
        stability_temp = np.diag(sensitivity03_probs_list_firsthalf[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens03_firsthalf.append(stability_temp)

    df_sleep_stability_all_sens03_firsthalf = pd.DataFrame(sleep_stability_list_sens03_firsthalf)
    df_sleep_stability_all_sens03_firsthalf = pd.concat([df_fif_files_subjects, df_sleep_stability_all_sens03_firsthalf], axis=1)
    print("Sensitivity 03 - first half")
    print(df_sleep_stability_all_sens03_firsthalf)

    # Sensitivity 03 - second half

    sleep_stability_list_sens03_secondhalf = []
    for i in range(len(sensitivity03_probs_list_secondhalf)):
        stability_temp = np.diag(sensitivity03_probs_list_secondhalf[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens03_secondhalf.append(stability_temp)

    df_sleep_stability_all_sens03_secondhalf = pd.DataFrame(sleep_stability_list_sens03_secondhalf)
    df_sleep_stability_all_sens03_secondhalf = pd.concat([df_fif_files_subjects, df_sleep_stability_all_sens03_secondhalf], axis=1)
    print("Sensitivity 03 - second half")
    print(df_sleep_stability_all_sens03_secondhalf)

    #average
    z = 0
    for s in probs_first:
        z = z + s
    first_average_probs = z / len(probs_first)
    print(first_average_probs)

    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5, 5))
    sns.heatmap(first_average_probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax,
                cmap='YlGnBu', annot=True, fmt='.3f', cbar_kws={"orientation": "horizontal", "fraction":0.1,
                                                                "label":"Transition Probability"})

    ax.set_xlabel("To sleep stage")
    ax.xaxis.tick_top()
    ax.set_ylabel("From sleep stage")
    ax.xaxis.set_label_position('top')
    plt.rcParams["figure.dpi"] = 150
    plt.show()

    #Spectrograms and Bandpowers
    first_hypnos = []
    second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep/' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        first_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1/30, data=first_mneraw_list[i]))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep/' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        second_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1 / 30, data=second_mneraw_list[i]))



    # Sensitivity_Analysis_02
    hypnoyp_temp_02 = []
    for i in range(len(hypnosensitivity02list_all)):
        hypnoyp_temp_02.append(yasa.hypno_upsample_to_data(hypnosensitivity02list_all[i], sf_hypno=1/30, data=sens_02_mnerawlist_all[i]))

    #sensitivity_analysis_3

    hypnolist = []
    firsthalf_hypnolist = []
    secondhalf_hypnolist = []
    for i in range(len(hypno_first_half_list)):
        HYPNOup_temp = yasa.hypno_upsample_to_data(hypno_list[i], sf_hypno=1/30, data=mne_raw_list[i])
        HYPNOup_temp4 = yasa.hypno_upsample_to_data(hypno_first_half_list[i], sf_hypno=1/30, data=sens03mnerawlist_first[i])
        HYPNOup_temp5 = yasa.hypno_upsample_to_data(hypno_second_half_list[i], sf_hypno=1/30, data=sens03mnerawlist_second[i])

        hypnolist.append(HYPNOup_temp)
        firsthalf_hypnolist.append(HYPNOup_temp4)
        secondhalf_hypnolist.append(HYPNOup_temp5)


    #example
    data = first_mneraw_list[1].get_data()
    chan = first_mneraw_list[1].ch_names
    sf = first_mneraw_list[1].info['sfreq']
    fig = yasa.plot_spectrogram(data[chan.index("F4_A1_fil")], sf, first_hypnos[1])
    plt.show()

    #bandpower
    bandpower_stages_list = []
    bandpower_second = []

    for i in range(len(mne_raw_list)):
        bandpower_stages_list.append(yasa.bandpower(mne_raw_list[i], hypno=hypnolist[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(bandpower_stages_list, keys=fif_files_subjects)
    print(df_bandpower_first)


    #Sensitivity Analysis 02
    sens02_bandpower_all = []
    for i in range(len(hypnosensitivity02list_all)):
        sens02_bandpower_all.append(yasa.bandpower(sens_02_mnerawlist_all[i], hypno=hypnoyp_temp_02[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(sens02_bandpower_all, keys=fif_files_subjects)
    print("Sensitivity Analysis 02")
    print(df_bandpower_first)

    #Sensitivity Analysis 03
    firsthalf_bandpowerlist = []
    secondhalf_bandpowerlist = []
    for i in range(len(firsthalf_hypnolist)):
        firsthalf_bandpowerlist.append(yasa.bandpower(sens03mnerawlist_first[i], hypno=firsthalf_hypnolist[i], include=(2,3,4)))
        secondhalf_bandpowerlist.append(yasa.bandpower(sens03mnerawlist_second[i], hypno=secondhalf_hypnolist[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(firsthalf_bandpowerlist, keys=fif_files_subjects)
    print("Sensitivity Analysis 03 - First Half")
    print(df_bandpower_first)

    df_bandpower_first = pd.concat(secondhalf_bandpowerlist, keys=fif_files_subjects)
    print("Sensitivity Analysis 03 - Second Half")
    print(df_bandpower_first)




@router.get("/sleep_analysis_sensitivity_luizaddsubject__final")
async def sleepanalysislouizsensitivityaddsubjectfinal(workflow_id: str,
                                                       step_id: str,
                                                       run_id: str):

    df_first_hypnos = []
    df_second_hypnos = []
    df_first_fif_files = []
    df_second_fif_files = []
    path_first = 'UU_Sleep_final/' + 'Group_1'
    for entries in os.listdir(path_first):
        if entries.endswith(".csv"):
            df_first_hypnos.append(entries)
        else:
            df_first_fif_files.append(entries)

    path_second = 'UU_Sleep_final/' + 'Group_2'
    for entries in os.listdir(path_second):
        if entries.endswith(".csv"):
            df_second_hypnos.append(entries)
        else:
            df_second_fif_files.append(entries)

    print(df_first_fif_files)
    print(df_first_hypnos)
    print(df_second_hypnos)
    print(df_second_fif_files)
    fif_files = df_first_fif_files + df_second_fif_files
    fif_files_subjects = []
    for i in range(len(fif_files)):
        fif_files_subjects.append(fif_files[i].split(".")[0])

    duration_first = []
    first_group_fif_files = []
    for entries in df_first_fif_files:
        path = 'UU_Sleep_final/Group_1' + entries
        data = mne.io.read_raw_fif(path)
        info = data.info
        raw_data = data.get_data()
        channels = data.ch_names
        duration_first.append(raw_data.shape[1]/info['sfreq'])

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        #first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        first_group_fif_files.append(df_signals)
        #print(df_signals)


    duration_second = []
    second_group_fif_files = []
    for entries in df_second_fif_files:
        path = 'UU_Sleep_final/Group_2' + entries
        data = mne.io.read_raw_fif(path)
        raw_data = data.get_data()
        channels = data.ch_names
        info = data.info
        duration_second.append(raw_data.shape[1] / info['sfreq'])

        list_signals = []
        for i in range(len(channels)):
            list_signals.append(raw_data[i])

        # first_group_fif_files.append(np.array(list_signals).T.tolist())
        df_signals = pd.DataFrame(np.array(list_signals).T.tolist(), columns=channels)
        second_group_fif_files.append(df_signals)
        # print(df_signals)

    info = mne.create_info(ch_names=channels, sfreq=info['sfreq'])
    array_list_first = []
    for i in range(len(first_group_fif_files)):
        array_list_first.append(first_group_fif_files[i].to_numpy())

    array_list_second = []
    for i in range(len(second_group_fif_files)):
        array_list_second.append(second_group_fif_files[i].to_numpy())

    first_mneraw_list = []
    second_mneraw_list = []
    for i in range(len(array_list_first)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_first[i]).T, info)
        first_mneraw_list.append(temp_MNEraw)

    for i in range(len(array_list_second)):
        temp_MNEraw = mne.io.RawArray(np.array(array_list_second[i]).T, info)
        second_mneraw_list.append(temp_MNEraw)

    ######## hypnogram
    mne_raw_list = first_mneraw_list + second_mneraw_list
    firsthalf_mneraw_list = first_mneraw_list + second_mneraw_list
    secondhalf_mneraw_list = first_mneraw_list + second_mneraw_list


    #print(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))
    #yasa.plot_hypnogram(np.squeeze(df.to_numpy()))

    list_first_hypnos = []
    list_second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep_final/Group_1' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        list_first_hypnos.append(np.squeeze(df.to_numpy()))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep_final/Group_2' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        list_second_hypnos.append(np.squeeze(df.to_numpy()))

    hypno_list = list_first_hypnos + list_second_hypnos

    ##########################################################################
    ##########################################################################
    ##########################################################################
    ### Sensitivity Analysis 02 - trim all files to 06h window

    Hypno_sensitivity02_list_first = []
    sens02mnerawlist_first = []
    for i in range(len(list_first_hypnos)):
        temp = list_first_hypnos[i][:720]
        Hypno_sensitivity02_list_first.append(temp)
        sens02mnerawlist_first.append(first_mneraw_list[i])
        if duration_first[i] > 21600:
            sens02mnerawlist_first[i].crop(tmin=0, tmax=21600)

    Hypno_sensitivity02_list_second = []
    sens02mnerawlist_second = []
    for i in range(len(list_second_hypnos)):
        temp = list_second_hypnos[i][:720]
        Hypno_sensitivity02_list_second.append(temp)
        sens02mnerawlist_second.append(second_mneraw_list[i])
        if duration_second[i] > 21600:
            sens02mnerawlist_second[i].crop(tmin=0, tmax=21600)

    sens_02_mnerawlist_all = sens02mnerawlist_first + sens02mnerawlist_second
    hypnosensitivity02_all = Hypno_sensitivity02_list_first + Hypno_sensitivity02_list_second

    ##### Sensitivity Analysis 03

    hypno_first_half_list = []
    hypno_second_half_list = []
    sens03mnerawlist_first = []
    sens03mnerawlist_second = []
    for i in range(len(mne_raw_list)):
        x = hypno_list[i].size/2
        temp_first = hypno_list[i][:int(x)]
        temp_second = hypno_list[i][int(x):]
        hypno_first_half_list.append(temp_first)
        hypno_second_half_list.append(temp_second)
        sens03mnerawlist_first.append(firsthalf_mneraw_list[i].copy().crop(tmin=0, tmax=int(x*30)))
        sens03mnerawlist_second.append(secondhalf_mneraw_list[i].copy().crop(tmin=int(x*30)))

    ### Sensitivity Analysis 02 - hypnograms

    hypnosensitivity02list_all = Hypno_sensitivity02_list_first + Hypno_sensitivity02_list_second
    sensitivity02_sleepstatistics = []
    for i in range(len(hypnosensitivity02list_all)):
        sensitivity02_sleepstatistics.append(yasa.sleep_statistics(hypnosensitivity02list_all[i], sf_hyp=1/30))

    df_sens02sleep_statistics = pd.DataFrame(sensitivity02_sleepstatistics)
    df_fif_files_subjects = pd.DataFrame(fif_files_subjects, columns=['subjects'])
    df_sens02sleep_statistics = pd.concat([df_fif_files_subjects, df_sens02sleep_statistics], axis=1)
    print('Sensitivity 02 - Sleep Statistics')
    print(df_sens02sleep_statistics)

    # Sensitivity Analysis 03 - Hypnograms

    sleepstats_firsthalf_list = []
    sleepstats_secondhalf_list = []
    for i in range(len(hypno_first_half_list)):
        temp_df1 = yasa.sleep_statistics(hypno_first_half_list[i], sf_hyp=1/30)
        temp_df2 = yasa.sleep_statistics(hypno_second_half_list[i], sf_hyp=1/30)

        sleepstats_firsthalf_list.append(temp_df1)
        sleepstats_secondhalf_list.append(temp_df2)

    df_sens03sleep_statisticsfirsthalf = pd.DataFrame(sleepstats_firsthalf_list)
    df_sens03sleep_statisticsfirsthalf = pd.concat([df_fif_files_subjects, df_sens03sleep_statisticsfirsthalf], axis=1)
    print('Sensitivity 03 - Sleep Statistics - first half list')
    print(df_sens03sleep_statisticsfirsthalf)

    df_sens03sleep_statisticssecondhalf = pd.DataFrame(sleepstats_secondhalf_list)
    df_sens03sleep_statisticssecondhalf = pd.concat([df_fif_files_subjects, df_sens03sleep_statisticssecondhalf], axis=1)
    print('Sensitivity 03 - Sleep Statistics - second half list')
    print(df_sens03sleep_statisticssecondhalf)

    ######################################################################################
    sleep_stats_first = []
    sleep_stats_second = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep_final/Group_1' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_first.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep_final/Group_2' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        sleep_stats_second.append(yasa.sleep_statistics(np.squeeze(df.to_numpy()), sf_hyp=1/30))

    df_first_sleep_statistics = pd.DataFrame(sleep_stats_first)
    df_second_sleep_statistics = pd.DataFrame(sleep_stats_second)

    first_fif_files_subjects = []
    for i in range(len(df_first_fif_files)):
        first_fif_files_subjects.append(df_first_fif_files[i].split(".")[0])

    df_first_fif_files_dataframe = pd.DataFrame(first_fif_files_subjects, columns = ['subjects'])

    second_fif_files_subjects = []
    for i in range(len(df_second_fif_files)):
        second_fif_files_subjects.append(df_second_fif_files[i].split(".")[0])

    df_second_fif_files_dataframe = pd.DataFrame(second_fif_files_subjects, columns=['subjects'])

    df_first_sleep_statistics = pd.concat([df_first_fif_files_dataframe, df_first_sleep_statistics], axis = 1)
    print('First folder - Sleep Statistics')
    print(df_first_sleep_statistics)

    df_second_sleep_statistics = pd.concat([df_second_fif_files_dataframe, df_second_sleep_statistics], axis = 1)
    print('Second folder - Sleep Statistics')
    print(df_second_sleep_statistics)

    ########################################################################
    #sleep transition matrix

    ## Sensitivity Analysis 02

    sensitivity02_counts_list = []
    sensitivity02_probs_list = []
    for i in range(len(hypnosensitivity02list_all)):
        counts, probs = yasa.transition_matrix(hypnosensitivity02list_all[i])
        sensitivity02_counts_list.append(counts)
        sensitivity02_probs_list.append(probs.round(3))

    # Sensitivity Analysis 03

    sensitivity03_counts_list_firsthalf = []
    sensitivity03_probs_list_firsthalf = []

    sensitivity03_counts_list_secondhalf = []
    sensitivity03_probs_list_secondhalf = []
    for i in range(len(hypno_first_half_list)):
        counts, probs = yasa.transition_matrix(hypno_first_half_list[i])
        sensitivity03_counts_list_firsthalf.append(counts)
        sensitivity03_probs_list_firsthalf.append(probs.round(3))

        counts, probs = yasa.transition_matrix(hypno_second_half_list[i])
        sensitivity03_counts_list_secondhalf.append(counts)
        sensitivity03_probs_list_secondhalf.append(probs.round(3))

    ################################################

    counts_first = []
    probs_first = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep_final/Group_1' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_first.append(counts)
        probs_first.append(probs.round(3))

    counts_second = []
    probs_second = []
    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep_final/Group_2' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        counts, probs = yasa.transition_matrix(np.squeeze(df.to_numpy()))
        counts_second.append(counts)
        probs_second.append(probs.round(3))

    #Concatenate lists
    Sleepmatrix_counts_list = counts_first + counts_second
    Sleepmatrix_probs_list = probs_first + probs_second

    #Worthless transitions
    worthless_transitions = []
    worthless_first_transitions = []
    worthless_second_transitions = []
    worthless_firsthalf_transitions = []
    worthless_secondhalf_transitions = []
    worthless_sens02_transitions = []

    for i in range(len(Sleepmatrix_counts_list)):
        temp = Sleepmatrix_counts_list[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_transitions.append(temp)

        temp = sensitivity02_counts_list[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_sens02_transitions.append(temp)

        temp = sensitivity03_counts_list_firsthalf[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_firsthalf_transitions.append(temp)

        temp = sensitivity03_counts_list_secondhalf[i] == 0
        temp.iloc[:, 0:] = temp.iloc[:, 0:].replace({True: 1, False: 0})
        worthless_secondhalf_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_transitions))

    print('Sensitivity 02')
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_sens02_transitions))

    print("Sensitivity 03 - first half")
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_firsthalf_transitions))

    print("Sensitivity 03 - second half")
    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_secondhalf_transitions))

    for i in range(len(counts_first)):
        temp = counts_first[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_first_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_first_transitions))

    for i in range(len(counts_second)):
        temp = counts_second[i] == 0
        temp.iloc[:,0:] = temp.iloc[:,0:].replace({True:1, False:0})
        worthless_second_transitions.append(temp)

    print(reduce(lambda x, y: x.add(y, fill_value=0), worthless_second_transitions))

    #probs
    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_first)):
        temp = probs_first[i][0:1]
        temp2 = probs_first[i][1:2]
        temp3 = probs_first[i][2:3]
        temp4 = probs_first[i][3:4]
        temp5 = probs_first[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=df_first_fif_files)
    df_N1_trans = pd.concat(N1_trans, keys=df_first_fif_files)
    df_N2_trans = pd.concat(N2_trans, keys=df_first_fif_files)
    df_N3_trans = pd.concat(N3_trans, keys=df_first_fif_files)
    df_REM_trans = pd.concat(REM_trans, keys=df_first_fif_files)

    first_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("first folder")
    print(first_probs_Sleep_Matrix)

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(probs_second)):
        temp = probs_second[i][0:1]
        temp2 = probs_second[i][1:2]
        temp3 = probs_second[i][2:3]
        temp4 = probs_second[i][3:4]
        temp5 = probs_second[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=df_second_fif_files)
    df_N1_trans = pd.concat(N1_trans, keys=df_second_fif_files)
    df_N2_trans = pd.concat(N2_trans, keys=df_second_fif_files)
    df_N3_trans = pd.concat(N3_trans, keys=df_second_fif_files)
    df_REM_trans = pd.concat(REM_trans, keys=df_second_fif_files)

    second_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("second folder")
    print(second_probs_Sleep_Matrix)

    # Sensitivity Analysis 02

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity02_probs_list)):
        temp = sensitivity02_probs_list[i][0:1]
        temp2 = sensitivity02_probs_list[i][1:2]
        temp3 = sensitivity02_probs_list[i][2:3]
        temp4 = sensitivity02_probs_list[i][3:4]
        temp5 = sensitivity02_probs_list[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=fif_files_subjects)
    df_N1_trans = pd.concat(N1_trans, keys=fif_files_subjects)
    df_N2_trans = pd.concat(N2_trans, keys=fif_files_subjects)
    df_N3_trans = pd.concat(N3_trans, keys=fif_files_subjects)
    df_REM_trans = pd.concat(REM_trans, keys=fif_files_subjects)

    sensitivity02_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("Sensitivity 02")
    print(sensitivity02_probs_Sleep_Matrix)

    # Sensitivity Analysis 03 - first half

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity03_probs_list_firsthalf)):
        temp = sensitivity03_probs_list_firsthalf[i][0:1]
        temp2 = sensitivity03_probs_list_firsthalf[i][1:2]
        temp3 = sensitivity03_probs_list_firsthalf[i][2:3]
        temp4 = sensitivity03_probs_list_firsthalf[i][3:4]
        temp5 = sensitivity03_probs_list_firsthalf[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=fif_files_subjects)
    df_N1_trans = pd.concat(N1_trans, keys=fif_files_subjects)
    df_N2_trans = pd.concat(N2_trans, keys=fif_files_subjects)
    df_N3_trans = pd.concat(N3_trans, keys=fif_files_subjects)
    df_REM_trans = pd.concat(REM_trans, keys=fif_files_subjects)

    sensitivity03_first_half_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("Sesnitivity Analysis - First half")
    print(sensitivity03_first_half_probs_Sleep_Matrix)

    # Sensitivity Analysis 03 - second half

    WAKE_trans = []
    N1_trans = []
    N2_trans = []
    N3_trans = []
    REM_trans = []

    for i in range(len(sensitivity03_probs_list_secondhalf)):
        temp = sensitivity03_probs_list_secondhalf[i][0:1]
        temp2 = sensitivity03_probs_list_secondhalf[i][1:2]
        temp3 = sensitivity03_probs_list_secondhalf[i][2:3]
        temp4 = sensitivity03_probs_list_secondhalf[i][3:4]
        temp5 = sensitivity03_probs_list_secondhalf[i][4:5]
        WAKE_trans.append(temp)
        N1_trans.append(temp2)
        N2_trans.append(temp3)
        N3_trans.append(temp4)
        REM_trans.append(temp5)

    df_WAKE_trans = pd.concat(WAKE_trans, keys=fif_files_subjects)
    df_N1_trans = pd.concat(N1_trans, keys=fif_files_subjects)
    df_N2_trans = pd.concat(N2_trans, keys=fif_files_subjects)
    df_N3_trans = pd.concat(N3_trans, keys=fif_files_subjects)
    df_REM_trans = pd.concat(REM_trans, keys=fif_files_subjects)

    sensitivity03_second_half_probs_Sleep_Matrix = pd.concat([df_WAKE_trans, df_N1_trans, df_N2_trans, df_N3_trans, df_REM_trans])
    print("Sensitivity Analysis 03 - Second Half")
    print(sensitivity03_second_half_probs_Sleep_Matrix)

    # Sleep fragmentation from probs

    sleep_stability_list = []
    for i in range(len(Sleepmatrix_probs_list)):
        stability_temp = np.diag(Sleepmatrix_probs_list[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list.append(stability_temp)

    df_sleep_stability_all = pd.DataFrame(sleep_stability_list)
    df_sleep_stability_all = pd.concat([df_fif_files_subjects, df_sleep_stability_all], axis=1)
    print('Both folders')
    print(df_sleep_stability_all)

    sleep_stability_list_first = []
    for i in range(len(probs_first)):
        stability_temp = np.diag(probs_first[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_first.append(stability_temp)

    df_sleep_stability_first = pd.DataFrame(sleep_stability_list_first)
    df_sleep_stability_first = pd.concat([df_first_fif_files_dataframe, df_sleep_stability_first], axis=1)
    print("First Folder")
    print(df_sleep_stability_first)

    sleep_stability_list_second = []
    for i in range(len(probs_second)):
        stability_temp = np.diag(probs_second[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_second.append(stability_temp)

    df_sleep_stability_second = pd.DataFrame(sleep_stability_list_second)
    df_sleep_stability_second = pd.concat([df_second_fif_files_dataframe, df_sleep_stability_second], axis=1)
    print("Second Folder")
    print(df_sleep_stability_second)

    # Sensitivity 02

    sleep_stability_list_sens02 = []
    for i in range(len(sensitivity02_probs_list)):
        stability_temp = np.diag(sensitivity02_probs_list[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens02.append(stability_temp)

    df_sleep_stability_sens02 = pd.DataFrame(sleep_stability_list_sens02)
    df_sleep_stability_sens02 = pd.concat([df_fif_files_subjects, df_sleep_stability_sens02], axis=1)
    print('Sensitivity 02')
    print(df_sleep_stability_sens02)

    # Sensitivity 03 - first half

    sleep_stability_list_sens03_firsthalf = []
    for i in range(len(sensitivity03_probs_list_firsthalf)):
        stability_temp = np.diag(sensitivity03_probs_list_firsthalf[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens03_firsthalf.append(stability_temp)

    df_sleep_stability_all_sens03_firsthalf = pd.DataFrame(sleep_stability_list_sens03_firsthalf)
    df_sleep_stability_all_sens03_firsthalf = pd.concat([df_fif_files_subjects, df_sleep_stability_all_sens03_firsthalf], axis=1)
    print("Sensitivity 03 - first half")
    print(df_sleep_stability_all_sens03_firsthalf)

    # Sensitivity 03 - second half

    sleep_stability_list_sens03_secondhalf = []
    for i in range(len(sensitivity03_probs_list_secondhalf)):
        stability_temp = np.diag(sensitivity03_probs_list_secondhalf[i].loc[2:, 2:]).mean().round(3)
        sleep_stability_list_sens03_secondhalf.append(stability_temp)

    df_sleep_stability_all_sens03_secondhalf = pd.DataFrame(sleep_stability_list_sens03_secondhalf)
    df_sleep_stability_all_sens03_secondhalf = pd.concat([df_fif_files_subjects, df_sleep_stability_all_sens03_secondhalf], axis=1)
    print("Sensitivity 03 - second half")
    print(df_sleep_stability_all_sens03_secondhalf)

    #average
    z = 0
    for s in probs_first:
        z = z + s
    first_average_probs = z / len(probs_first)
    print(first_average_probs)

    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5, 5))
    sns.heatmap(first_average_probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax,
                cmap='YlGnBu', annot=True, fmt='.3f', cbar_kws={"orientation": "horizontal", "fraction":0.1,
                                                                "label":"Transition Probability"})

    ax.set_xlabel("To sleep stage")
    ax.xaxis.tick_top()
    ax.set_ylabel("From sleep stage")
    ax.xaxis.set_label_position('top')
    plt.rcParams["figure.dpi"] = 150
    plt.show()

    #Spectrograms and Bandpowers
    first_hypnos = []
    second_hypnos = []
    for i in range(len(df_first_hypnos)):
        path = 'UU_Sleep_final/Group_1' + str(df_first_hypnos[i])
        df = pd.read_csv(path)
        first_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1/30, data=first_mneraw_list[i]))

    for i in range(len(df_second_hypnos)):
        path = 'UU_Sleep_final/Group_2' + str(df_second_hypnos[i])
        df = pd.read_csv(path)
        second_hypnos.append(yasa.hypno_upsample_to_data(np.squeeze(df.to_numpy()), sf_hypno=1 / 30, data=second_mneraw_list[i]))



    # Sensitivity_Analysis_02
    hypnoyp_temp_02 = []
    for i in range(len(hypnosensitivity02list_all)):
        hypnoyp_temp_02.append(yasa.hypno_upsample_to_data(hypnosensitivity02list_all[i], sf_hypno=1/30, data=sens_02_mnerawlist_all[i]))

    #sensitivity_analysis_3

    hypnolist = []
    firsthalf_hypnolist = []
    secondhalf_hypnolist = []
    for i in range(len(hypno_first_half_list)):
        HYPNOup_temp = yasa.hypno_upsample_to_data(hypno_list[i], sf_hypno=1/30, data=mne_raw_list[i])
        HYPNOup_temp4 = yasa.hypno_upsample_to_data(hypno_first_half_list[i], sf_hypno=1/30, data=sens03mnerawlist_first[i])
        HYPNOup_temp5 = yasa.hypno_upsample_to_data(hypno_second_half_list[i], sf_hypno=1/30, data=sens03mnerawlist_second[i])

        hypnolist.append(HYPNOup_temp)
        firsthalf_hypnolist.append(HYPNOup_temp4)
        secondhalf_hypnolist.append(HYPNOup_temp5)


    #example
    data = first_mneraw_list[1].get_data()
    chan = first_mneraw_list[1].ch_names
    sf = first_mneraw_list[1].info['sfreq']
    fig = yasa.plot_spectrogram(data[chan.index("F4_A1_fil")], sf, first_hypnos[1])
    plt.show()

    #bandpower
    bandpower_stages_list = []
    bandpower_second = []

    for i in range(len(mne_raw_list)):
        bandpower_stages_list.append(yasa.bandpower(mne_raw_list[i], hypno=hypnolist[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(bandpower_stages_list, keys=fif_files_subjects)
    print(df_bandpower_first)


    #Sensitivity Analysis 02
    sens02_bandpower_all = []
    for i in range(len(hypnosensitivity02list_all)):
        sens02_bandpower_all.append(yasa.bandpower(sens_02_mnerawlist_all[i], hypno=hypnoyp_temp_02[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(sens02_bandpower_all, keys=fif_files_subjects)
    print("Sensitivity Analysis 02")
    print(df_bandpower_first)

    #Sensitivity Analysis 03
    firsthalf_bandpowerlist = []
    secondhalf_bandpowerlist = []
    for i in range(len(firsthalf_hypnolist)):
        firsthalf_bandpowerlist.append(yasa.bandpower(sens03mnerawlist_first[i], hypno=firsthalf_hypnolist[i], include=(2,3,4)))
        secondhalf_bandpowerlist.append(yasa.bandpower(sens03mnerawlist_second[i], hypno=secondhalf_hypnolist[i], include=(2,3,4)))

    df_bandpower_first = pd.concat(firsthalf_bandpowerlist, keys=fif_files_subjects)
    print("Sensitivity Analysis 03 - First Half")
    print(df_bandpower_first)

    df_bandpower_first = pd.concat(secondhalf_bandpowerlist, keys=fif_files_subjects)
    print("Sensitivity Analysis 03 - Second Half")
    print(df_bandpower_first)
