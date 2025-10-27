import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from string import punctuation

data = pd.read_excel("../data/raw/data_raw.xlsx")

new_data = data.drop(["Вид", "ID", "Кампания"], axis = 1)
new_data = new_data.dropna()
new_data = new_data.rename(columns={
    "Комментарии": "text",
    "Эмоциональная окраска": "type",
})
data = new_data['type'] = new_data['type'].map({
    'Мусор': 0,
    'Негативная': 1,
    'Нейтральная': 2,
    'Позитивная': 3,
    })

tbl = str.maketrans("", "", punctuation)

s = new_data['text'].str.lower()
s = s.str.translate(tbl)
s = s.str.replace(r'\s+', ' ', regex=True)
s = s.str.strip()

new_data['label'] = s

data_count = new_data.groupby("type")["label"].count()
df_count = data_count.to_frame()

sns.barplot(x=df_count.index, y = df_count['label'], palette = 'summer')

plt.title("Кол-во комментариев разных классов")
plt.xlabel("Окрас")
plt.ylabel("Кол-во")
plt.legend()
plt.savefig("analyze.svg")

X = new_data["text"]
y = new_data["type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, stratify=y, shuffle=True
)

pd.concat([X_train, y_train], axis=1).to_csv("../data/output/train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("../data/output/test.csv")