from fastapi import APIRouter, Query
from statsmodels.graphics.tsaplots import acf, pacf
from scipy import signal
import mne

router = APIRouter()

# region EEG Function pre-processing and functions
data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)


# endregion


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
async def partial_autocorrelation(input_name: str,
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

# Estimate welch
@router.get("/return_periodogram", tags=["return_periodogram"])
# Validation is done inline in the input of the function
async def estimate_periodogram(input_name: str,
                               input_window: str | None = Query("hann",
                                                              regex="^(boxcar)$|^(triang)$|^(blackman)$|^(hamming)$|^(hann)$|^(bartlett)$|^(flattop)$|^(parzen)$|^(bohman)$|^(blackmanharris)$|^(nuttall)$|^(barthann)$|^(cosine)$|^(exponential)$|^(tukey)$|^(taylor)$"),
                               input_nfft: int | None = 256,
                               input_return_onesided: bool | None = True,
                               input_scaling: str | None = Query("density", regex="^(density)$|^(spectrum)$"),
                               input_axis: int | None = -1) -> dict:
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    for i in range(len(channels)):
        if input_name == channels[i]:
            f, pxx_den = signal.periodogram(raw_data[i], info['sfreq'], window=input_window,
                                            nfft=input_nfft, return_onesided=input_return_onesided, scaling=input_scaling,
                                            axis=input_axis)
            return {'frequencies': f.tolist(), 'power spectral density': pxx_den.tolist()}
    return {'Channel not found'}
