import os
from datetime import datetime
import json
import pingouin as pg
import seaborn as sns
import math
import yasa
from yasa import plot_spectrogram, spindles_detect, sw_detect, SleepStaging
import paramiko
from fastapi import APIRouter, Query
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, lfilter, sosfilt, freqs, freqs_zpk, sosfreqz
from statsmodels.graphics.tsaplots import acf, pacf
from scipy import signal
from scipy.integrate import simps
from pmdarima.arima import auto_arima
import seaborn as seaborn
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

from app.utils.utils_eeg import load_data_from_edf, load_file_from_local_or_interim_edfbrowser_storage
from app.utils.utils_general import validate_and_convert_peaks, validate_and_convert_power_spectral_density, \
    create_notebook_mne_plot, get_neurodesk_display_id, get_annotations_from_csv, create_notebook_mne_modular, \
    get_single_file_from_local_temp_storage, get_local_storage_path, get_local_edfbrowser_storage_path, \
    get_single_file_from_edfbrowser_interim_storage

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import mne
import requests
from yasa import spindles_detect
from pyedflib import highlevel
from app.pydantic_models import *

router = APIRouter()

# region EEG Function pre-processing and functions
# TODO Finalise the use of file dynamically
data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)
NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

# data = mne.io.read_raw_fif("/neurodesktop-storage/trial_av_processed.fif")

#data = mne.io.read_raw_edf("example_data/psg1 anonym2.edf", infer_types=True)

# endregion

def rose_plot(ax, angles, bins=12, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)

    fig = plt.figure(1)

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

    html_str = mpld3.fig_to_html(fig)
    return html_str

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
async def list_channels(step_id: str,
                        run_id: str,
                        file_used: str | None = Query("original",
                                        regex="^(original)$|^(printed)$"),
                        ) -> dict:

    # If file is altered we retrieve it from the edf interim storage fodler
    if file_used == "printed":
        path_to_storage = get_local_edfbrowser_storage_path(run_id, step_id)
        name_of_file = get_single_file_from_edfbrowser_interim_storage(run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)
    else:
        # If not we use it from the directory input files are supposed to be
        path_to_storage = get_local_storage_path(run_id, step_id)
        name_of_file = get_single_file_from_local_temp_storage(run_id, step_id)
        data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    channels = data.ch_names
    return {'channels': channels}


@router.get("/return_autocorrelation", tags=["return_autocorrelation"])
# Validation is done inline in the input of the function
async def return_autocorrelation(step_id: str, run_id: str,
                                 input_name: str, input_adjusted: bool | None = False,
                                 input_qstat: bool | None = False, input_fft: bool | None = False,
                                 input_bartlett_confint: bool | None = False,
                                 input_missing: str | None = Query("none",
                                                                   regex="^(none)$|^(raise)$|^(conservative)$|^(drop)$"),
                                 input_alpha: float | None = None, input_nlags: int | None = None,
                                 file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                 ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)

    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            z = acf(raw_data[i], adjusted=input_adjusted, qstat=input_qstat,
                    fft=input_fft,
                    bartlett_confint=input_bartlett_confint,
                    missing=input_missing, alpha=input_alpha,
                    nlags=input_nlags)

            to_return = {}
            # Parsing the results of acf into a single object
            # Results will change depending on our input
            if input_qstat and input_alpha:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['confint'] = z[1].tolist()
                to_return['qstat'] = z[2].tolist()
                to_return['pvalues'] = z[3].tolist()
            elif input_qstat:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['qstat'] = z[1].tolist()
                to_return['pvalues'] = z[2].tolist()
            elif input_alpha:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['confint'] = z[1].tolist()
            else:
                to_return['values_autocorrelation'] = z.tolist()

            print("RETURNING VALUES")
            print(to_return)
            return to_return
    return {'Channel not found'}


@router.get("/return_partial_autocorrelation", tags=["return_partial_autocorrelation"])
# Validation is done inline in the input of the function
async def return_partial_autocorrelation(step_id: str, run_id: str,
                                         input_name: str,
                                         input_method: str | None = Query("none",
                                                                          regex="^(none)$|^(yw)$|^(ywadjusted)$|^(ywm)$|^(ywmle)$|^(ols)$|^(ols-inefficient)$|^(ols-adjusted)$|^(ld)$|^(ldadjusted)$|^(ldb)$|^(ldbiased)$|^(burg)$"),
                                         input_alpha: float | None = None, input_nlags: int | None = None,
                                         file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                         ) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)

    # path_to_storage = get_local_storage_path(run_id, step_id)
    # name_of_file = get_single_file_from_local_temp_storage(run_id, step_id)
    # data = load_data_from_edf(path_to_storage + "/" + name_of_file)

    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            z = pacf(raw_data[i], method=input_method, alpha=input_alpha, nlags=input_nlags)

            to_return = {}
            # Parsing the results of acf into a single object
            # Results will change depending on our input
            if input_alpha:
                to_return['values_partial_autocorrelation'] = z[0].tolist()
                to_return['confint'] = z[1].tolist()
            else:
                to_return['values_partial_autocorrelation'] = z.tolist()

            print("RETURNING VALUES")
            print(to_return)
            return to_return
    return {'Channel not found'}


@router.get("/return_filters", tags=["return_filters"])
# Validation is done inline in the input of the function besides
async def return_filters(
                         step_id: str, run_id: str,
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
    path_to_storage = get_local_storage_path(run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(run_id, step_id)
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
# Validation is done inline in the input of the function
async def estimate_welch(
                        step_id: str, run_id: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)

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
            return {'frequencies': f.tolist(), 'power spectral density': pxx_den.tolist()}
    return {'Channel not found'}

@router.get("/return_stft", tags=["return_stft"])
# Validation is done inline in the input of the function
async def estimate_stft(
                        step_id: str, run_id: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
            plt.show()

            html_str = mpld3.fig_to_html(fig)
            plt.savefig(get_local_storage_path(step_id, run_id) + "/output/" + 'plot.png')
            to_return["figure"] = html_str
            return to_return
    return {'Channel not found'}


# Find peaks
@router.get("/return_peaks", tags=["return_peaks"])
# Validation is done inline in the input of the function
async def return_peaks(step_id: str, run_id: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
            to_return = {}
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

            html_str = mpld3.fig_to_html(fig)
            to_return["figure"] = html_str
            # Html_file = open("index.html", "w")
            # Html_file.write(html_str)
            # Html_file.close()
            # print(to_return)
            return to_return
    return {'Channel not found'}


# Estimate welch
@router.get("/return_periodogram", tags=["return_periodogram"])
# Validation is done inline in the input of the function
async def estimate_periodogram(step_id: str, run_id: str,input_name: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
# TODO TMIN and TMAX probably should be removed
async def return_power_spectral_density(step_id: str, run_id: str,input_name: str,
                                        tmin: float | None = None,
                                        tmax: float | None = None,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
async def calculate_alpha_delta_ratio(step_id: str, run_id: str,input_name: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
            low, high = 8, 12

            # Find intersecting values in frequency vector
            idx_alpha = np.logical_and(freqs >= low, freqs <= high)
            freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

            # Compute the absolute power by approximating the area under the curve
            alpha_power = simps(psd[idx_alpha], dx=freq_res)
            #################################################

            #
            low, high = 0.5, 4

            # Find intersecting values in frequency vector
            idx_05_4 = np.logical_and(freqs >= low, freqs <= high)

            # Compute the absolute power by approximating the area under the curve
            delta_power = simps(psd[idx_05_4], dx=freq_res)

            return {'alpha_delta_ratio': alpha_power/delta_power}


@router.get("/return_asymmetry_indices", tags=["return_asymmetry_indices"])
async def calculate_asymmetry_indices(step_id: str, run_id: str,input_name_1: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
async def calculate_alpha_variability(step_id: str, run_id: str,input_name: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
async def return_predictions(step_id: str, run_id: str,input_name: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)

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

# Spindles detection
@router.get("/spindles_detection")
async def detect_spindles(step_id: str,
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)


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
                plt.savefig(get_local_storage_path(step_id, run_id) + "/output/" + 'plot.png')

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

# Slow Waves detection
@router.get("/slow_waves_detection")
async def detect_slow_waves(
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
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)

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
                plt.savefig(get_local_storage_path(step_id, run_id) + "/output/" + 'plot.png')

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
async def sleep_statistics_hypnogram(sampling_frequency: float | None = Query(default=1/30)):

    hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')

    df = pd.DataFrame.from_dict(sleep_statistics(list(hypno['stage']), sf_hyp=sampling_frequency), orient='index', columns=['value'])

    return{'sleep statistics':df.to_json(orient='split')}

@router.get("/sleep_transition_matrix")
async def sleep_transition_matrix():
    #fig = plt.figure(1)
    #ax = plt.subplot(111)

    to_return = {}
    fig = plt.figure(1)

    hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')

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
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return["figure"] = html_str

    return{'Counts transition matrix (number of transitions from stage A to stage B).':counts.to_json(orient='split'),
           'Conditional probability transition matrix, i.e. given that current state is A, what is the probability that the next state is B.':probs.to_json(orient='split'),
           'figure': to_return}

@router.get("/sleep_stability_extraction")
async def sleep_stability_extraction():

    hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')

    counts, probs = yasa.transition_matrix(list(hypno['stage']))

    return{'stability of sleep stages':np.diag(probs.loc[2:, 2:]).mean().round(3)}

@router.get("/spectrogram_yasa")
async def spectrogram_yasa(name: str,
                           current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):

    data = mne.io.read_raw_fif("example_data/XX_Firsthalf_raw.fif")
    hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')
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
            fig = plt.figure(1)
            fig = yasa.plot_spectrogram(array_data, sf, hypno, cmap='Spectral_r')
            plt.show()

            html_str = mpld3.fig_to_html(fig)
            to_return["figure"] = html_str

            return {'Figure': to_return}
    return {'Channel not found'}

@router.get("/bandpower_yasa")
async def bandpower_yasa(name: str,
                         current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):

    data = mne.io.read_raw_fif("example_data/XX_Firsthalf_raw.fif")
    hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    print(channels)
    sf = info['sfreq']

    hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)

    df = yasa.bandpower(data, hypno=hypno)

    return {'DataFrame':df.to_json(orient='split')}

@router.get("/spindles_detect_two_dataframes")
async def spindles_detect_two_dataframes(current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):

    data = mne.io.read_raw_fif("example_data/XX_Firsthalf_raw.fif")
    hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    sf = info['sfreq']
    hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)

    sp = yasa.spindles_detect(data, hypno=hypno, include=(0,1,2,3))
    if sp!=None:
        df_1 = sp.summary()
        df_2 = sp.summary(grp_chan=True, grp_stage=True)

        to_return = {}
        fig = plt.figure(1)
        fig = sp.plot_average(center='Peak', time_before=1, time_after=1)
        plt.show()
        html_str = mpld3.fig_to_html(fig)
        to_return["figure"] = html_str

        return {'DataFrame_1':df_1.to_json(orient='split'), 'DataFrame_2':df_2.to_json(orient='split'),'Figure':to_return}
    else:
        return {'No spindles detected'}

@router.get("/sw_detect_two_dataframes")
async def sw_detect_two_dataframes(current_sampling_frequency_of_the_hypnogram: float | None = Query(default=1/30)):

    #data = mne.io.read_raw_fif("example_data/XX_Firsthalf_raw.fif")
    #hypno = pd.read_csv('example_data/XX_Firsthalf_Hypno.csv')
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    sf = info['sfreq']
    #hypno = yasa.hypno_upsample_to_data(list(hypno['stage']), sf_hypno=current_sampling_frequency_of_the_hypnogram, data=data)

    sw = yasa.sw_detect(data, coupling=True,remove_outliers=True)
    if sw!=None:
        df_1 = sw.summary()
        df_2 = sw.summary(grp_chan=True, grp_stage=True)

        to_return = {}
        ax = plt.subplot(projection='polar')
        figure_2 = rose_plot(ax, df_1['PhaseAtSigmaPeak'], density=False, offset=0, lab_unit='degrees', start_zero=False)
        to_return['figure_2'] = figure_2


        fig = plt.figure(1)
        pg.plot_circmean(df_1['PhaseAtSigmaPeak'])
        print('Circular mean: %.3f rad' % pg.circ_mean(df_1['PhaseAtSigmaPeak']))
        print('Vector length: %.3f' % pg.circ_r(df_1['PhaseAtSigmaPeak']))
        plt.show()
        html_str = mpld3.fig_to_html(fig)
        to_return["figure"] = html_str

        return {'DataFrame_1':df_1.to_json(orient='split'), 'DataFrame_2':df_2.to_json(orient='split'),'Figure':to_return,
                'Circular mean (rad):': pg.circ_mean(df_1['PhaseAtSigmaPeak']),
                'Vector length (rad):': pg.circ_r(df_1['PhaseAtSigmaPeak'])}
    else:
        return {'No slow-waves detected'}











# Spindles detection
# Annotations_to_add have the folowing format which follows the format of adding it to the file with mne
# [ [starts], [durations], [names]  ]
@router.get("/save_annotation_to_file")
async def save_annotation_to_file(step_id: str,
                          run_id: str,
                          name: str,
                          annotations_to_add: str,
                          file_used: str | None = Query("original", regex="^(original)$|^(printed)$")):
    # Open file
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)
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
async def mne_open_eeg(step_id: str, run_id: str, current_user: str | None = None) -> dict:
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
    path_to_storage = get_local_storage_path(run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(run_id, step_id)
    file_full_path = path_to_storage + "/" + name_of_file

    # Give permissions in working folder
    channel.send("sudo chmod a+rw /home/user/neurodesktop-storage/runtime_config/run_" + run_id + "_step_" + step_id +"/edfbrowser_interim_storage\n")

    # Opening EDFBrowser
    channel.send("cd /home/user/neurodesktop-storage/runtime_config/run_" + run_id + "_step_" + step_id +"/edfbrowser_interim_storage\n")
    # print("/home/user/EDFbrowser/edfbrowser /home/user/'" + file_full_path + "'\n")
    channel.send("/home/user/EDFbrowser/edfbrowser '/home/user" + file_full_path + "'\n")

    # OLD VISUAL STUDIO CODE CALL and terminate
    # channel.send("pkill -INT code -u user\n")
    # channel.send("/neurocommand/local/bin/mne-1_0_0.sh\n")
    # channel.send("nohup /usr/bin/code -n /home/user/neurodesktop-storage/created_1.ipynb --extensions-dir=/opt/vscode-extensions --disable-workspace-trust &\n")


# TODO chagne parameter name
@router.get("/return_signal", tags=["return_signal"])
# Start date time is returned as miliseconds epoch time
async def return_signal(step_id: str, run_id: str,input_name: str) -> dict:
    path_to_storage = get_local_storage_path(run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(run_id, step_id)
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
async def mne_return_annotations(step_id: str, run_id: str, file_name: str | None = "annotation_test.csv") -> dict:
    # Default value proable isnt needed in final implementation
    annotations = get_annotations_from_csv(file_name)
    return annotations





@router.post("/receive_notebook_and_selection_configuration", tags=["receive__notebook_and_selection_configuration"])
async def receive_notebook_and_selection_configuration(input_config: ModelNotebookAndSelectionConfiguration,step_id: str, run_id: str,file_used: str | None = Query("original", regex="^(original)$|^(printed)$")) -> dict:
    # TODO TEMP
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)

    # data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)

    raw_data = data.get_data(return_times=True)

    print(input_config)
    # Produce new notebook
    create_notebook_mne_modular(file_to_save="created_1",
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
        data.save( NeurodesktopStorageLocation + "/trial_av_processed.fif", "all", overwrite = True, buffer_size_sec=None)
    else:
        data.save(NeurodesktopStorageLocation + "/trial_av_processed.fif", "all", overwrite = True, buffer_size_sec=None)

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
@router.get("/test/montage", tags=["test_montage"])
async def test_montage() -> dict:
    raw_data = data.get_data()
    info = data.info
    print('\nBUILT-IN MONTAGE FILES')
    print('======================')
    print(info)
    print(raw_data)
    ten_twenty_montage = mne.channels.make_standard_montage('example_data/trial_av')
    print(ten_twenty_montage)

    # create_notebook_mne_plot("hello", "again")



@router.get("/test/notebook", tags=["test_notebook"])
# Validation is done inline in the input of the function
async def test_notebook(input_test_name: str, input_slices: str,
                        ) -> dict:
    create_notebook_mne_plot("hello", "again")



@router.get("/envelope_trend", tags=["envelope_trend"])
# Validation is done inline in the input of the function
async def return_envelopetrend(
                               step_id: str,
                               run_id: str,
                               input_name: str,
                               window_size: int | None = None,
                               percent: float | None = None,
                               input_method: str | None = Query("none", regex="^(Simple)$|^(Cumulative)$|^(Exponential)$"),
                               file_used: str | None = Query("original", regex="^(original)$|^(printed)$")) -> dict:
    data = load_file_from_local_or_interim_edfbrowser_storage(file_used, run_id, step_id)
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
