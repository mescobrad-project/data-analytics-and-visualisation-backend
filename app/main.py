import mne
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.graphics.tsaplots import acf

tags_metadata = [
    {
        "name": "test_return_autocorrelation",
        "description": "Test of the test_return_autocorrelation function with visual representation.",
        "externalDocs": {
            "description": "Test",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "test_list_channels",
        "description": "Test of the test_list_channels function with visual representation.",
        "externalDocs": {
            "description": "Test",
            "url": "https://www.google.com/",
        }
    }
]

app = FastAPI(openapi_tags=tags_metadata)

# region CORS Setup
# This region enables FastAPI's built in CORSMiddleware allowing cross-origin requests allowing communication with
# the React front end
origins = [
    "http://localhost:3000",
    "http://localhost:3000/auto_correlation",
    "localhost:3000"
    "localhost:3000/auto_correlation"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# endregion


# region TEST EEG example data import and handling
data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types=True)


# end region

# region Routes of the application
@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello World"}


todos = [
    {
        "id": "1",
        "item": "Read a book."
    },
    {
        "id": "2",
        "item": "Cycle around town."
    }
]


@app.get("/test/list/channels", tags=["test_list_channels"])
async def test_list_channels() -> dict:
    channels = data.ch_names
    # print(channels)
    # print(type(channels))
    return {'channels': channels}


@app.get("/test/return_autocorrelation", tags=["test_return_autocorrelation"])
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


@app.get("/test")
async def test():
    return {'this is a test'}
# endregion
