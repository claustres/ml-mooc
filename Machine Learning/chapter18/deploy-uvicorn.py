from predict import Data
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.post('/predict')
def predict(payload: dict):
  data = Data(payload)
  return { 'y': data.predict() }
