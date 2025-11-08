# TF-IDF

## О системе

Система представляет собой учебный проект классификации комментариев. Для этого были использованы модели классификации на основе TF-IDF и logreg, а также несколько вариантов ruBert.

## Функционал
Система представлет собой скрипты для обучени моделей, скрипты и бэкенд и фронтенд для их использования, а также базу данных на postgresql для сохранения запросов.

## Структура
```
project-root/
├─ README.md # описание проекта
├─ requirements.txt # зависимости Python
├─ notebooks/ # jupyter ноутбуки для исследований
│ └─ bert.ipynb
├─ scripts/ # основная логика
│ ├─ backend.py
│ ├─ frontend.py
│ ├─ test.py # тест моделей из pickle
│ └─ test_checkpoint.py # тест моделей из checkpoints
├─ models/ # сохранённые модели
├─ data/ # исходные и промежуточные данные
└─ env.example # пример переменных окружения
```
## Использование

1. Скопируйте содержимое env.example в файл .env и настройте (при необходимости) переменные среды
2. Создайте окружение и загрузите зависимости:
```
python venv -m venv
venv/Scripts/Activate
pip install -r requirements.txt
cd scripts
```

3. Если нужно, обучите модели:
```
python classifier.py // для TF-IDF + logreg
python classifier.py // для GridSearch
python bert.py // для ruBert
```

4. Запустите систему:
```
docker-compose up --build
python backend.py
streamlit run frontend.py
```

Готово!