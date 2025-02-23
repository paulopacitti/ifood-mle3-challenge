from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
from config import MODEL_PATH, DEVICE


class InputText(BaseModel):
    text: str


class OutputPrediction(BaseModel):
    label: str
    score: float


class Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            device=DEVICE,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
        )

    def predict(self, inputText: InputText) -> List[OutputPrediction]:
        return self.pipeline(inputText.text)[0]
