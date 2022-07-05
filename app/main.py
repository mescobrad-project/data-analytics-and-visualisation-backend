from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import routers_eeg, routers_mri
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .utils.utils_general import create_neurodesk_user, read_all_neurodesk_users, \
    save_neurodesk_user

tags_metadata = [
    {
        "name": "return_autocorrelation",
        "description": "return_autocorrelation function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_partial_autocorrelation",
        "description": "return_partial_autocorrelation function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_filters",
        "description": "return_filters function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "list_channels",
        "description": "test_list_channels function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_welch",
        "description": "return_welch function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_stft",
        "description": "return_stft function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_peaks",
        "description": "return_peaks function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_power_spectral_density",
        "description": "return_power_spectral_density function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_periodogram",
        "description": "return_periodogram function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
    {
        "name": "return_spindles_detection",
        "description": "return_spindles_detection function with visual representation.",
        "externalDocs": {
            "description": "-",
            "url": "https://www.google.com/",
        }
    },
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

app.mount("/static", StaticFiles(directory="/neurodesktop-storage"), name="static")

# endregion


# region Routes of the application
@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello World"}


@app.get("/test/chart", tags=["root"])
async def root():
    return FileResponse('index.html')


@app.get("/test/read/users", tags=["root"])
async def test_read_users():
    # Test write user in local storage

    read_all_neurodesk_users()
    return "Success"

@app.get("/test/write/user", tags=["root"])
async def test_write_user(name, password):
    # Test write user in local storage

    save_neurodesk_user(name, password)
    return "Success"

@app.get("/test/add/user", tags=["root"])
async def test_add_user(name, password):
    # Must add user both at ubuntu and at file of guacamole
    # 1 - Adding  at apache guacamole - needs sudo privileges
    # file etc/guacamole/user
    # 2 - Adding user at ubuntu
    # Done with ssh

    create_neurodesk_user(name, password)
    return "Success"

# Include routers from other folders
app.include_router(routers_eeg.router)
app.include_router(routers_mri.router)


# endregion
