from transformers import pipeline

path = "../models/checkpoint-5613"

pipe = pipeline("text-classification", model=path, tokenizer=path, device_map="auto")

while True:
    text = input("Введите текст: ")
    result = pipe(text)
    print(result)