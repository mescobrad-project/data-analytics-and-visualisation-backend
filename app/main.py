from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import mne
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
    "localhost:3000"
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
data = mne.io.read_raw_edf("example_data/trial_av.edf", infer_types = True)


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


@app.post("/test/return_autocorrelation", tags=["test_return_autocorrelation"])
async def test_return_autocorrelation(name: dict) -> dict:
    print("Starting AutoCorellation")
    raw_data = data.get_data()
    channels = data.ch_names
    for i in range(len(channels)):
        if name["name"] == channels[i]:
            z = acf(raw_data[i])
            print("RETURNING VALUES")
            return {'values_autocorrelation': z.tolist()}
    return {'Channel not found'}


@app.get("/test")
async def test():
    return {'this is a test'}
# endregion