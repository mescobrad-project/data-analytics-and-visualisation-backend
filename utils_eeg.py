from typing import Optional
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, Path
import mne
import numpy as np
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from statsmodels.graphics.tsaplots import acf
from scipy.integrate import simps
from scipy import signal
from mne.time_frequency import psd_array_multitaper
import yasa
from yasa import plot_spectrogram, spindles_detect, sw_detect, SleepStaging
from scipy.signal import butter, lfilter, freqz

app = FastAPI()
data = mne.io.read_raw_edf("trial_av.edf", infer_types = True)

def butter_lowpass(cutoff, fs, type_filter, order=5):
    if type_filter != 'bandpass':
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=type_filter, analog=False)
        return b, a
    else:
        nyq = 0.5 * fs
        low = cutoff[0]/nyq
        high = cutoff[1]/nyq
        b, a = butter(order, [low, high], btype=type_filter, analog=False)
        return b, a

def butter_lowpass_filter(data, cutoff, fs, type_filter, order=5):
    b, a = butter_lowpass(cutoff, fs, type_filter, order=order)
    y = lfilter(b, a, data)
    return y

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = signal.welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def calcsmape(actual, forecast):
    return 1/len(actual) * np.sum(2 * np.abs(forecast-actual) / (np.abs(actual) + np.abs(forecast)))

class InsertChannel(BaseModel):
    channel: str

#Return list of channels
@app.get("/get_list_of_channels")
def read_mne():
    channels = data.ch_names
    return {'channels': channels}

#Return one specific channel
@app.get("/return_specific_channel")
def return_channel_time_series(name: str):
    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            return raw_data[i].tolist()
    return {'Channel not found'}

#Return predictions using ARIMA
@app.get("/return_predictions")
def return_predictions(name: str, test_size: int):
    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            data_channel = raw_data[i]
            train, test = data_channel[:-test_size], data_channel[-test_size:]
            x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], data_channel.shape[0]))
            model = auto_arima(train, start_p=1, start_q=1,
                               test='adf',
                               max_p=5, max_q=5,
                               m=1,
                               d=1,
                               seasonal=False,
                               start_P=0,
                               D=None,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
            prediction, confint = model.predict(n_periods=test_size, return_conf_int=True)
            smape = calcsmape(test, prediction)
            example = model.summary()
            results_as_html = example.tables[0].as_html()
            df_0 = pd.read_html(results_as_html, header=0, index_col=0)[0]

            results_as_html = example.tables[1].as_html()
            df_1 = pd.read_html(results_as_html, header=0, index_col=0)[0]

            results_as_html = example.tables[2].as_html()
            df_2 = pd.read_html(results_as_html, header=0, index_col=0)[0]
            return {'predictions': prediction.tolist(), 'error': smape, 'confint': confint, 'first_table':df_0.to_json(orient="split"), 'second table':df_1.to_json(orient="split"), 'third table':df_2.to_json(orient="split")}
    return {'Channel not found'}

# Return Autocorrelation
@app.get("/return_autocorrelation")
def return_autocorrelation(name: str):
    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            z = acf(raw_data[i])
            return {'values_autocorrelation': z.tolist()}
    return {'Channel not found'}

# Estimate welch
@app.get("/estimate_welch")
def estimate_welch(name: str):
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            f, Pxx_den = signal.welch(raw_data[i], info['sfreq'], nperseg=4 * info['sfreq'])
            return {'frequencies': f.tolist(), 'power spectral density': Pxx_den.tolist()}
    return {'Channel not found'}

# Estimate periodogram
@app.get("/estimate_periodogram")
def estimate_periodogram(name: str):
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            f, Pxx_den = signal.periodogram(raw_data[i], info['sfreq'])
            return {'frequencies': f.tolist(), 'power spectral density': Pxx_den.tolist()}
    return {'Channel not found'}

# Estimate stft
@app.get("/estimate_stft")
def estimate_stft(name: str, nperseg: int):
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            f, t, Zxx = signal.stft(raw_data[i], info['sfreq'], nperseg=nperseg)
            return {'frequencies': f.tolist(), 'STFT': np.abs(Zxx).tolist(), 'array of segment times': t.tolist()}
    return {'Channel not found'}

# Estimate multitaper
@app.get("/estimate_multitaper")
def estimate_multitaper(name: str):
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            psd, freqs = psd_array_multitaper(raw_data[i], info['sfreq'], adaptive=True, normalization='full', verbose=0)
            return {'frequencies': freqs.tolist(), 'psd': psd.tolist()}
    return {'Channel not found'}

# Spindles detection
@app.get("/spindles_detection")
def detect_spindles(name: str):
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    list_all = []
    for i in range(len(channels)):
        if name == channels[i]:
            sp = spindles_detect(raw_data[i] * 1e6, info['sfreq'])
            df = sp.summary()
            for i in range(len(df)):
                list_start_end = []
                start = df.iloc[i]['Start'] * info['sfreq']
                end = df.iloc[i]['End'] * info['sfreq']
                list_start_end.append(start)
                list_start_end.append(end)
                list_all.append(list_start_end)
            return {'detected spindles': list_all}
    return {'Channel not found'}

# Slow Waves detection
@app.get("/slow_waves_detection")
def detect_slow_waves(name: str):
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    list_all = []
    for i in range(len(channels)):
        if name == channels[i]:
            sw = sw_detect(raw_data[i] * 1e6, info['sfreq'])
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

# Detect peaks
@app.get("/peaks_detection")
def detect_peaks(name: str):
    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            peaks = signal.find_peaks(raw_data[i])
            return {'detected peaks': peaks[0].tolist()}
    return {'Channel not found'}

# Relative/Absolute ratio
@app.get("/ratio_between_two_frequency_bands")
def compute_ratio_between_two_frequency_bands(name: str, band_1: str, band_2: str):
    raw_data = data.get_data()
    info = data.info
    sf = info['sfreq']
    channels = data.ch_names
    for i in range(len(channels)):
        if name == channels[i]:
            data_1 = raw_data[i]
            print('no')
            if band_1 == "delta":
                distribution_1 = [0.5, 4]
            if band_2 == "beta":
                distribution_2 = [12, 30]
            # Define the duration of the window to be 4 seconds
            win_sec = 4

            # Delta/beta ratio based on the absolute power
            db = bandpower(data_1, sf, distribution_1, win_sec) / bandpower(data_1, sf, distribution_2, win_sec)

            # Delta/beta ratio based on the relative power
            db_rel = bandpower(data_1, sf, distribution_1, win_sec, True) / bandpower(data_1, sf, distribution_2, win_sec, True)

            return {'Delta div beta ratio (absolute):' : db, 'Delta div beta ratio (relative)': db_rel}
    return {'Channel not found'}

# Return relative bandpower per channel on the whole recording
@app.get("/relative_bandpower_per_channel")
def relative_bandpower_per_channel():
    raw_data = data.get_data()
    channels = data.ch_names
    info = data.info
    df = yasa.bandpower(raw_data, sf=info['sfreq'], ch_names=channels)
    json_file = df.to_json(orient="split")
    return json_file

# Apply lowpass or highpass filter
@app.get("/lowpass_highpass_filter")
def apply_lowpass_highpass(name: str, filter: str, cutoff: int, order: int):
    raw_data = data.get_data()
    channels = data.ch_names
    info = data.info
    for i in range(len(channels)):
        if name == channels[i]:
            data_1 = raw_data[i]
            y = butter_lowpass_filter(data_1, cutoff, info['sfreq'], filter, order)
            return {'filtered signal': y.tolist()}
    return {'Channel not found'}

# Apply bandpass filter
@app.get("/bandpass_filter")
def apply_bandpass(name: str, filter: str, low: int, high:int, order: int):
    raw_data = data.get_data()
    channels = data.ch_names
    info = data.info
    for i in range(len(channels)):
        if name == channels[i]:
            data_1 = raw_data[i]
            y = butter_lowpass_filter(data_1, [low, high], info['sfreq'], filter, order)
            return {'filtered signal': y.tolist()}
    return {'Channel not found'}