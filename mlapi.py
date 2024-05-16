from fastapi import FastAPI,File,UploadFile
import numpy as np
import PIL.Image
import PIL.ImageOps
import io
import pickle
import os
import pandas as pd

os.system("cls")

app = FastAPI()

with open('foodnutrition_baru.pkl','rb') as f:
    model = pickle.load(f)

class_names = ['Ayam Bakar','Ayam Geprek',
 'Ayam Goreng','Ayam Tepung','Bakso','Chicken Katsu',
 'Donat','Gado-Gado', 'Kopi', 'Mie Ayam', 'Mie Instan',
'Nasi Goreng','Pecel Lele','Rendang','Sate','Sop','Soto',
 'Telur Balado']

def getCalorie():
    nutrisi_csv = pd.read_csv('nutrisi_origin.csv')
    Kalori = nutrisi_csv['Kalori']
    Protein = nutrisi_csv['Protein']
    Lemak = nutrisi_csv['Lemak']

    return Kalori,Protein,Lemak
    

@app.get('/')
async def hello():
    return "hello"

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img = await file.read()
    img = PIL.Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize(size=(244,244))
    img = np.array(img)
    img = np.expand_dims(img,axis=0)
    preds = model.predict(img)
    predicted_label = class_names[np.argmax(preds[0])]

    Kalori,Protein,Lemak = getCalorie()
  
    # return f"Prediction : {np.argmax(preds[0])}"

    return {"Prediction" : predicted_label}

# img = PIL.Image.open(io.BytesIO(img)).convert('RGB')
# img = img.resize(size=(224,224))
# img = np.array(img)
# img = np.expand_dims(img,axis=0)
# preds = model.predict(img)
# print(preds)