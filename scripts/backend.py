from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from transformers import pipeline

STANDART_PATH = "../models/checkpoint-5613"

class RequestBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
    model: str = Field(STANDART_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Все работает!"}

@app.post("/predict")
def root(payload: RequestBody = Body(..., embed=True)):
    path = payload.model
    pipe = pipeline("text-classification", model=path, tokenizer=path, device_map="auto")
    result = int(pipe(payload.text))
    variants = ["Мусор", "Негативный отзыв", "Нейтральный отзыв", "Позитивный отзыв"]
    return {"message": f"Результат: {variants[result]}"}