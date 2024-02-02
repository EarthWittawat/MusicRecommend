from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import algorithm # all data processing and clustering algorithms will be in this file
import uvicorn

## allow CORS
origins = ['*']
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def home():
    msg = {
        'message': 'Music Recommendation',
        'api_endpoints': ['music']
    }
    result = JSONResponse(content=msg)
    return result
@app.get("/api/music")
async def cluster(lyrics: str):
    msg = {
        'message' : 'song',
        'cluster' : algorithm.get_music(lyrics, return_id=False),
    }
    result = JSONResponse(content=msg)
    return result
