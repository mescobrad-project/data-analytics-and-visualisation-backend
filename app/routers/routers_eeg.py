from datetime import datetime
import json
import math
import yasa
import paramiko
from fastapi import APIRouter, Query
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, lfilter, sosfilt, freqs, freqs_zpk, sosfreqz
from statsmodels.graphics.tsaplots import acf, pacf
from scipy import signal
from scipy.integrate import simps
from pmdarima.arima import auto_arima
import mne
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import lxml
from app.utils.utils_general import validate_and_convert_peaks, validate_and_convert_power_spectral_density, \
    create_notebook_mne_plot, get_neurodesk_display_id, get_annotations_from_csv, create_notebook_mne_modular

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
# data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)

data = mne.io.read_raw_fif("/neurodesktop-storage/trial_av_processed.fif")

#data = mne.io.read_raw_edf("example_data/psg1 anonym2.edf", infer_types=True)

# endregion

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
async def list_channels() -> dict:
    channels = data.ch_names
    return {'channels': channels}


@router.get("/return_autocorrelation", tags=["return_autocorrelation"])
# Validation is done inline in the input of the function
async def return_autocorrelation(input_name: str, input_adjusted: bool | None = False,
                                 input_qstat: bool | None = False, input_fft: bool | None = False,
                                 input_bartlett_confint: bool | None = False,
                                 input_missing: str | None = Query("none",
                                                                   regex="^(none)$|^(raise)$|^(conservative)$|^(drop)$"),
                                 input_alpha: float | None = None, input_nlags: int | None = None) -> dict:
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
async def return_partial_autocorrelation(input_name: str,
                                         input_method: str | None = Query("none",
                                                                          regex="^(none)$|^(yw)$|^(ywadjusted)$|^(ywm)$|^(ywmle)$|^(ols)$|^(ols-inefficient)$|^(ols-adjusted)$|^(ld)$|^(ldadjusted)$|^(ldb)$|^(ldbiased)$|^(burg)$"),
                                         input_alpha: float | None = None, input_nlags: int | None = None) -> dict:
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
async def return_filters(input_name: str,
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
                         input_fs_freq: float | None = None
                         ) -> dict:
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
async def estimate_welch(input_name: str,
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
                         input_average: str | None = Query("mean", regex="^(mean)$|^(median)$")) -> dict:
    data.crop(tmin=tmin, tmax=tmax)
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
async def estimate_stft(input_name: str,
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
                         input_axis: int | None = -1) -> dict:
    data.crop(tmin=tmin, tmax=tmax)
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
            to_return["figure"] = html_str
            return to_return
    return {'Channel not found'}


# Find peaks
@router.get("/return_peaks", tags=["return_peaks"])
# Validation is done inline in the input of the function
async def return_peaks(input_name: str,
                       input_height=None,
                       input_threshold=None,
                       input_distance: int | None = None,
                       input_prominence=None,
                       input_width=None,
                       input_wlen: int | None = None,
                       input_rel_height: float | None = None,
                       input_plateau_size=None) -> dict:
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
async def estimate_periodogram(input_name: str,
                               tmin: float | None = 0,
                               tmax: float | None = None,
                               input_window: str | None = Query("hann",
                                                                regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                               input_nfft: int | None = 256,
                               input_return_onesided: bool | None = True,
                               input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                               input_axis: int | None = -1) -> dict:
    data.crop(tmin=tmin, tmax=tmax)
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


# Return power_spectral_density
@router.get("/return_power_spectral_density", tags=["return_power_spectral_density"])
# Validation is done inline in the input of the function
async def return_power_spectral_density(input_name: str,
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
                                        input_verbose: str | None = None
                                        ) -> dict:
    data.crop(tmin=tmin, tmax=tmax)
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
async def calculate_alpha_delta_ratio(input_name: str,
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
                                      input_average: str | None = Query("mean", regex="^(mean)$|^(median)$")) -> dict:
    data.crop(tmin=tmin, tmax=tmax)
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


@router.get("/asymmetry_indices")
async def calculate_asymmetry_indices(input_name_1: str,
                                      input_name_2: str,
                                      input_window: str | None = Query("hann",
                                                          regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                                      input_nperseg: int | None = 256,
                                      input_noverlap: int | None = None,
                                      input_nfft: int | None = None,
                                      input_return_onesided: bool | None = True,
                                      input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                                      input_axis: int | None = -1,
                                      input_average: str | None = Query("mean", regex="^(mean)$|^(median)$")) -> dict:

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

@router.get("/alpha_variability")
async def calculate_alpha_variability(input_name: str,
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
                                      input_average: str | None = Query("mean", regex="^(mean)$|^(median)$")) -> dict:
    data.crop(tmin=tmin, tmax=tmax)
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

@router.get("/return_predictions")
async def return_predictions(name: str,
                             test_size: int,
                             future_seconds: int,
                             start_p: int | None = 1,
                             start_q: int | None = 1,
                             max_p: int | None = 5,
                             max_q: int | None = 5,
                             method: str | None = Query("lbfgs",
                                                        regex="^(lbfgs)$|^(newton)$|^(nm)$|^(bfgs)$|^(powell)$|^(cg)$|^(ncg)$|^(basinhopping)$"),
                             information_criterion: str | None = Query("aic",
                                                                       regex="^(aic)$|^(bic)$|^(hqic)$|^(oob)$") ):
    raw_data = data.get_data()
    channels = data.ch_names
    info = data.info
    sampling_frequency = info['sfreq']
    for i in range(len(channels)):
        if name == channels[i]:
            data_channel = raw_data[i]
            train, test = data_channel[:-test_size], data_channel[-test_size:]
            #x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], data_channel.shape[0]))
            model = auto_arima(train, start_p=start_p, start_q=start_q,
                               test='adf',
                               max_p=max_p, max_q=max_q,
                               m=1,
                               d=1,
                               seasonal=False,
                               start_P=0,
                               D=None,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True,
                               method=method,
                               information_criterion=information_criterion)
            prediction, confint = model.predict(n_periods=test_size, return_conf_int=True)
            smape = calcsmape(test, prediction)
            example = model.summary()
            results_as_html = example.tables[0].as_html()
            df_0 = pd.read_html(results_as_html, header=0, index_col=0)[0]

            results_as_html = example.tables[1].as_html()
            df_1 = pd.read_html(results_as_html, header=0, index_col=0)[0]

            results_as_html = example.tables[2].as_html()
            df_2 = pd.read_html(results_as_html, header=0, index_col=0)[0]

            z = future_seconds*sampling_frequency

            prediction, confint = model.predict(n_periods=int(z), return_conf_int=True)
            return {'predictions': prediction.tolist(), 'error': smape, 'confint': confint, 'first_table':df_0.to_json(orient="split"), 'second table':df_1.to_json(orient="split"), 'third table':df_2.to_json(orient="split")}
    return {'Channel not found'}



@router.get("/mne/open/eeg", tags=["mne_open_eeg"])
# Validation is done inline in the input of the function
# Slices are send in a single string and then de
async def mne_open_eeg(input_run_id: str, input_step_id: str, current_user: str | None = None) -> dict:
    # # Create a new jupyter notebook with the id of the run and step for recognition
    # create_notebook_mne_plot(input_run_id, input_step_id)

    # Initiate ssh connection with neurodesk container
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("neurodesktop", 22, username="user", password="password")
    channel = ssh.invoke_shell()
    display_id = get_neurodesk_display_id()
    channel.send("export DISPLAY=" + display_id + "\n")
    # Close previous isntances of code for the user
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! THIS USER MUST CHANGE TO CURRENTLY USED USER
    channel.send("pkill -INT code -u user\n")

    channel.send("/neurocommand/local/bin/mne-1_0_0.sh\n")
    channel.send("nohup /usr/bin/code -n /home/user/neurodesktop-storage/created_1.ipynb --extensions-dir=/opt/vscode-extensions --disable-workspace-trust &\n")
    # channel.send("nohup code &\n")
    # channel.send("nohup code /home/user/neurodesktop-storage/TestEEG.ipynb --extensions-dir=/opt/vscode-extensions &\n")
    # channel.send(
    #     "nohup recon-all -subject " + input_test_name + " -i " + input_file + " -all > freesurfer_log.txtr &\n")
    #


@router.get("/return_signal", tags=["return_signal"])
# Start date time is returned as miliseconds epoch time
async def return_signal(input_name: str) -> dict:
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
async def mne_return_annotations(file_name: str | None = "annotation_test.csv") -> dict:
    # Default value proable isnt needed in final implementation
    annotations = get_annotations_from_csv(file_name)
    return annotations





@router.post("/receive_notebook_and_selection_configuration", tags=["receive__notebook_and_selection_configuration"])
async def receive_notebook_and_selection_configuration(input_config: ModelNotebookAndSelectionConfiguration) -> dict:
    # TODO TEMP
    data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)

    raw_data = data.get_data(return_times=True)

    print(input_config)
    # Produce new notebook
    create_notebook_mne_modular(file_to_save="created_1", file_to_open="trial_av.edf", notches_enabled=input_config.notches_enabled, notches_length= input_config.notches_length, annotations=True, bipolar_references=input_config.bipolar_references, reference_type= input_config.type_of_reference,
                                reference_channels_list=input_config.channels_reference,selection_start_time= input_config.selection_start_time,selection_end_time= input_config.selection_end_time)

    # If there is a selection channel we need to crop
    if input_config.selection_channel != "":
        # data.crop(float(input_config.selection_start_time), float(input_config.selection_end_time))

        # data.save("/neurodesktop-storage/trial_av_processed.fif", "all", float(input_config.selection_start_time), float(input_config.selection_end_time), overwrite = True, buffer_size_sec=24)
        data.save("/neurodesktop-storage/trial_av_processed.fif", "all", overwrite = True, buffer_size_sec=None)
    else:
        data.save("/neurodesktop-storage/trial_av_processed.fif", "all", overwrite = True, buffer_size_sec=None)

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
# Slices are send in a single string and then de
async def test_notebook(input_test_name: str, input_slices: str,
                        ) -> dict:
    create_notebook_mne_plot("hello", "again")

