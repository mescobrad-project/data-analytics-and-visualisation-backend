import math

from fastapi import APIRouter, Query
from scipy.signal import butter, lfilter, sosfilt, freqs, freqs_zpk, sosfreqz
from statsmodels.graphics.tsaplots import acf, pacf
from scipy import signal
import mne
import matplotlib.pyplot as plt
import mpld3
import numpy as np

router = APIRouter()

# region EEG Function pre-processing and functions
data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)


# endregion

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
                    frequency_response_output = sosfreqz(filter_created, worN=input_worn, whole=input_whole )
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
                         input_window: str | None = Query("hann",
                                                          regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                         input_nperseg: int | None = 256,
                         input_noverlap: int | None = None,
                         input_nfft: int | None = 256,
                         input_return_onesided: bool | None = True,
                         input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                         input_axis: int | None = -1,
                         input_average: str | None = Query("mean", regex="^(mean)$|^(median)$")) -> dict:
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
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

