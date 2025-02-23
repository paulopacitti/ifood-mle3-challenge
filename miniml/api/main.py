from typing import Union
from fastapi import FastAPI
from model import Model, InputText

model = Model()
app = FastAPI()


@app.post("/predict")
def predict(request: InputText):
    return model.predict(request)
