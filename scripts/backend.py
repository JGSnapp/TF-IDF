from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import uvicorn
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
load_dotenv()
import pickle

STANDART_PATH = os.getenv("STANDART_PATH", "../models/checkpoint-5613")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://app:secret@localhost:5432/mydb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def insert_request(engine, comment: str, comment_type: int, model: str):
    sql = text("""
        INSERT INTO requests (comment, comment_type, model)
        VALUES (:comment, :comment_type, :model)
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "comment": comment,
            "comment_type": comment_type,
            "model": model,
        })


def get_requests(engine):
    sql = text("""
        SELECT id, comment, comment_type, model, created_at
        FROM requests
        ORDER BY created_at DESC
    """)
    with engine.connect() as conn:
        res = conn.execute(sql)
        return res.mappings().all()


class RequestBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
    model: str = Field(STANDART_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Все работает!"}

@app.get("/get_list")
def get_list():
    try:
        return get_requests(engine)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(payload: RequestBody = Body(...)):
    model_path = payload.model
    text_input = payload.text
    if model_path[-3:] == "pkl":
        gs = pickle.load(open(model_path, 'rb'))
        result = int(gs.predict([text_input]))
    else:
        pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)
        result = int(pipe(text_input)[0]['label'])
    variants = ["Мусор", "Негативный отзыв", "Нейтральный отзыв", "Позитивный отзыв"]
    insert_request(engine, comment = text_input, comment_type = result, model = model_path)
    return {"message": f"Результат: {variants[result]}"}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=1000, reload=True)