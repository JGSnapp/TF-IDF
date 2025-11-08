import streamlit as st
import requests

st.title("Оценка настроения комментария")

predict_url = "http://localhost:1000/predict"
get_list_url = "http://localhost:1000/get_list"

text = st.text_area("Введите комментарий:", height=150)
model_path = st.text_input("Путь/ID модели (опционально)", value="../models/checkpoint-5613")

placeholder = st.empty()

if st.button("Отправить"):
    with st.spinner("Отправляю запрос..."):
            try:
                r = requests.post(predict_url, json={"text": text, "model": model_path}, timeout=60)
                data = r.json()
                placeholder.success(data.get("message", "OK"))
            except Exception as e:
                placeholder.error(f"Не удалось отправить запрос: {e}")

r = requests.get(get_list_url, timeout=60)
r.raise_for_status()
data = r.json()
st.header("История запросов")
for item in data:
    st.write(f"- Комментарий: {item['comment']}, Тип: {item['comment_type']}, Модель: {item['model']}")