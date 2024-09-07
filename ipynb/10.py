# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/10.ipynb#scrollTo=zV__984yWw5r" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# Устанавливаем необходимые библиотеки
# !pip install -q seaborn
# !pip install scikit-learn
# !pip install xgboost

# %%
import os
import tempfile
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier

# %% [markdown]
# ---
#
# # Задание 1
#
# ---

# %%
# Загрузка набора данных
# !wget https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv

# %%
data = pd.read_csv("../assets/fake_news.csv")
data = data.dropna()
data["text"] = data["title"] + "\n" + data["text"]

# %%
print(data.info())

# %%
# Разделим данные на обучающие и тестовые множества
x_train, x_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=7
)


# %%
mapping = {"REAL": 1, "FAKE": 0}
for k, v in mapping.items():
    x = np.mean(y_train == k)
    y = np.mean(y_test == k)
    print(f"{k} train: {x}, test: {y}")


# %%
# Преобразуем текстовые данные в числовые признаки с помощью TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)

# %%
# Обучим модель с использованием PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# %%
# Сделаем предсказания и оценим точность модели
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100, 2)}%")

# %%
# Построим и визуализируем матрицу ошибок
conf_mat = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"],
)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

# %% [markdown]
# ---
#
# # Задание 2
#
# ---

# %%
# Загрузка набора данных
# !wget https://storage.yandexcloud.net/academy.ai/practica/parkinsons.data

# %%
try:
    data = pd.read_csv("./parkinsons.data")
except FileNotFoundError:
    data = pd.read_csv("../assets/parkinsons.data")

data = data.dropna()
# print(data.head())
# print(data.info())
# print(data.columns)
data.head()

# %%
data.describe()

# %%
sns.pairplot(data=data[data.columns[0:24]])
plt.show()

# %%
data_numeric = data.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 5))
sns.heatmap(data_numeric.corr(), annot=True, cmap="cubehelix_r", annot_kws={"size": 8}, fmt="0.1f")
plt.show()

# %%
# Разделение данных на признаки и целевую переменную
X = data.drop(["name", "status"], axis=1)
y = data["status"]

# %%
# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# %%
# Стандартизация данных
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# %%
# Создание и обучение модели XGBoost
model = XGBClassifier()
model.fit(x_train, y_train)

# %%
# Предсказания на тестовой выборке
y_pred = model.predict(x_test)

# %%
# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy * 100, 2)}%")

# %% [markdown]
# ---
#
# # Задание 3
#
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 140, "referenced_widgets": ["d875ad9a5fb147f6ab3a57f0cb9bddec", "8bd4b6c07fb54a76a8b319119cbc0e34", "d44efc429f414826bace9295f2517a8a", "955e30d1d7c84086aa0eae579d055e37", "cf7d0c6a205540898c03a2227216a4ff", "3e0e95e0ef2446a7ad5b8493e46c7061", "9541246ec1c84b79b3655edde0c744cf", "e2d3c9531e154ed2b588f8f9ebba46d7", "69cecd42b1634f808e4c2d2830a4314c", "bb2a3aea760c40a7b5def7c0d33785ec", "ce5b36e934a6431399aa1adc522f4aba"]} id="YHTGHqWNBRSz" outputId="5c696d00-aa51-49ef-f6f1-914103d99960"
# Загрузка набора данных groove
_train_dataset, _val_dataset, _test_dataset = tfds.load(
    "groove/full-16000hz", split=["train", "validation", "test"]
)


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="7ZwBAurvCceX" outputId="8b4336e4-adbc-4eb8-9e98-27e0af1b750a"
def extract_bpm(dataset):
    bpm_values = []
    for data in dataset:
        bpm = data["bpm"].numpy()
        bpm_values.append(bpm)
    return bpm_values


bpm_train = extract_bpm(_train_dataset)

plt.hist(bpm_train, bins=20, edgecolor="black")
plt.title("BPM Distribution in Training Dataset")
plt.xlabel("BPM")
plt.ylabel("Frequency")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="JNRW8ibarhY1" outputId="0740475b-a0c9-4a5d-fd2b-4ff78d8d6612"
cnt_bpm_dict = Counter(bpm_train)
print(cnt_bpm_dict)


# %% id="J3K7pTb1C4KO"
def preprocess(data):
    # Преобразование аудио в спектрограмму
    audio = data["audio"]
    spectrogram = tf.signal.stft(audio, frame_length=256, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    # Добавление одного измерения
    spectrogram = tf.expand_dims(spectrogram, axis=-1)

    # Приведение спектрограмм к фиксированному размеру
    spectrogram = tf.image.resize(spectrogram, [128, 129])

    return spectrogram, data["bpm"]


# %% id="nEsRRwQzDNb8"
train_dataset = (
    _train_dataset.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
)
val_dataset = (
    _val_dataset.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
)
test_dataset = (
    _test_dataset.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
)

# %% id="TdGA2IATZ05G"
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(128, 129, 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

# %% id="Z8MKTWa_E3pA"
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# %% colab={"base_uri": "https://localhost:8080/"} id="-fiNnzKEE9Bg" outputId="473266c2-b546-4022-c3b9-0de9938b2ce0"
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100)

# %% colab={"base_uri": "https://localhost:8080/"} id="hLYXaQ12vn57" outputId="449ae576-315c-452f-c534-797627fe2b66"
print(history.history["loss"][-1])

# %% colab={"base_uri": "https://localhost:8080/"} id="cAiqoIuKv4rQ" outputId="ab167702-76ff-4ad9-be22-e07241a5e065"
while history.history["loss"][-1] > 1:
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="6TRkhbviFDaP" outputId="f73dfa19-474b-42ff-957b-55d3cc6c5061"
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Ошибка перекрестной энтропии")
plt.xlabel("Эпохи")
plt.ylabel("Ошибки")
plt.legend()
plt.show()


# %% id="Dhn0P89hLW9k"
def get_bpm_predictions(dataset):
    y_true = []
    y_pred = []
    for spectrogram, bpm in dataset:
        preds = model.predict(spectrogram)
        y_true.extend(bpm.numpy())
        y_pred.extend(np.round(preds))  # Округление предсказанных значений
    return y_true, y_pred


# %% colab={"base_uri": "https://localhost:8080/"} id="w-QKSkqTLeXg" outputId="a4a98b6e-42fd-4457-d6b2-270abddea08d"
y_true, y_pred = get_bpm_predictions(test_dataset)


# %% colab={"base_uri": "https://localhost:8080/"} id="uhH42istQ-iK" outputId="15072fbf-138a-432b-c949-11a5e6a4c6bd"
def accuracy_per_bpm(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    bpm_values = np.unique(y_true)
    accuracy_dict = {}

    for bpm in bpm_values:
        idx = np.where(y_true == bpm)
        true_bpm = np.array(y_true)[idx]
        pred_bpm = np.array(y_pred_rounded)[idx]
        accuracy = np.mean(abs(true_bpm - pred_bpm) < 5)
        count = len(true_bpm)
        accuracy_dict[bpm] = {"accuracy": accuracy, "count": count}

    return accuracy_dict


accuracy_dict = accuracy_per_bpm(y_true, y_pred)

for bpm, accuracy in accuracy_dict.items():
    print(
        f'BPM: {bpm}, Accuracy: {accuracy["accuracy"]:.2f}, Track count: {accuracy["count"]}',
    )
