import mne
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.graphics.tsaplots import acf, pacf
from .routers import routers_eeg

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

# Include routers from other folders
app.include_router(routers_eeg.router)

# endregion
