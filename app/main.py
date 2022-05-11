from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import routers_eeg

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


# region Routes of the application
@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello World"}


# Include routers from other folders
app.include_router(routers_eeg.router)

# endregion
