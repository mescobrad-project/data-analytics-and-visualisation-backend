from fastapi import APIRouter, Query
from statsmodels.graphics.tsaplots import acf, pacf
import mne

router = APIRouter()

data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)

@router.get("/return_partial_autocorrelation", tags=["test_return_partial_autocorrelation"])
# Validation is done inline in the input of the function
async def return_partial_autocorrelation(input_name: str,
                                         input_method: str | None = Query("none",
                                                                    regex="^(none)$|^(yw)$|^(ywadjusted)$|^(ywm)$|^(ywmle)$|^(ols)$|^(ols-inefficient)$|^(ols-adjusted)$|^(ld)$|^(ldadjusted)$|^(ldb)$|^(ldbiased)$|^(burg)$"),
                                         input_alpha: float | None = None, input_nlags: int | None = None) -> dict:
    # print("Starting AutoCorellation")
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
                # to_return['alpha'] = z[2].tolist()
            else:
                to_return['values_partial_autocorrelation'] = z.tolist()

            print("RETURNING VALUES")
            print(to_return)
            return to_return
    return {'Channel not found'}

@router.get("/test/list/channels", tags=["test_list_channels"])
async def test_list_channels() -> dict:
    channels = data.ch_names
    # print(channels)
    # print(type(channels))
    return {'channels': channels}


@router.get("/test/return_autocorrelation", tags=["test_return_autocorrelation"])
# Validation is done inline in the input of the function
async def test_return_autocorrelation(input_name: str, input_adjusted: bool | None = False,
                                      input_qstat: bool | None = False, input_fft: bool | None = False,
                                      input_bartlett_confint: bool | None = False,
                                      input_missing: str | None = Query("none",
                                                                        regex="^(none)$|^(raise)$|^(conservative)$|^(drop)$"),
                                      input_alpha: float | None = None, input_nlags: int | None = None) -> dict:
    # print("Starting AutoCorellation")
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
                # to_return['confint'] = z[2].tolist()
            elif input_qstat:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['qstat'] = z[1].tolist()
                to_return['pvalues'] = z[2].tolist()
            elif input_alpha:
                to_return['values_autocorrelation'] = z[0].tolist()
                to_return['confint'] = z[1].tolist()
                # to_return['alpha'] = z[2].tolist()
            else:
                to_return['values_autocorrelation'] = z.tolist()

            print("RETURNING VALUES")
            print(to_return)
            return to_return
    return {'Channel not found'}
