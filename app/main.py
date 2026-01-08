from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.core import predict_food

app = FastAPI(title="Food Recognition API")

@app.get("/")
def root():
    return {"status": "Backend is running"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = predict_food(image)
    return result

