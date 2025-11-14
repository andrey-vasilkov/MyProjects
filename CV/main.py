import numpy as np

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title='Symbol detection', docs_url='/docs')

# load model
model = Model()



@app.get("/status")
def status():
    return "Кажется, работает"
# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    image = np.array(list(map(int, image[1:-1].split(','))))
    pred = model.predict(image)
    return {'prediction': pred}

# static files
app.mount('/static', StaticFiles(directory='static', html=True), name='static')
