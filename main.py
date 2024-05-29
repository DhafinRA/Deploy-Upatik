# special thanks to chatgpt
from typing import Union
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
import my_module

app = FastAPI()

# CSV file path
csv_file_path = "class_data.csv"
label_mapping = my_module.read_label_mappings_from_csv(csv_file_path)

# Template folder path
templates = Jinja2Templates(directory=".")\

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict(file: UploadFile):
    contents = await file.read()

    image_base64 = my_module.image_to_base64(contents)

    response = httpx.post('http://localhost:8601/v1/models/my_simple_model:predict',
                   json=image_base64)

    predictions = response.json().get('predictions', [])

    translated_predictions = []

    for prediction in predictions:
        max_index = prediction.index(max(prediction))
        translated_prediction = label_mapping[max_index]
        translated_predictions.append(translated_prediction)

    return {"predictions": translated_predictions}

@app.get("/version")
async def check_model():
    response= httpx.get('http://localhost:8601/v1/models/my_simple_model/versions/1')

    return response.json()