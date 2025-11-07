import streamlit as st
import requests

st.title("Оценка настроения комментария")

api_url = "http://localhost:1000/predict"

text = st.text_area("Введите комментарий:", height=150)
model_path = st.text_input("Путь/ID модели (опционально)", value="../models/checkpoint-5613")

placeholder = st.empty()

if st.button("Отправить"):
    with st.spinner("Отправляю запрос..."):
            try:
                r = requests.post(api_url, json={"text": text, "model": model_path}, timeout=60)
                data = r.json()
                placeholder.success(data.get("message", "OK"))
                st.json(data)
            except Exception as e:
                placeholder.error(f"Не удалось отправить запрос: {e}")