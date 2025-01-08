# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---


# %% [markdown] id="W-cexaowCTp5"
# 1. На 3 балла. Обучите модель с точностью не менее 90% предсказывать сарказм в новостных заголовках. Составьте 5 произвольных заголовков, которых нет в датасете и проверьте на них обученную модель, сделайте выводы. Ссылка на [датасет](https://storage.yandexcloud.net/academy.ai/Sarcasm_Headlines_Dataset_v2.json.zip).
# 2. На 4 балла. Используйте [русский корпус новостей от Lenta.ru](https://www.kaggle.com/datasets/yutkin/corpus-of-russian-news-articles-from-lenta/data) подберите и обучите модель классифицировать новости по заголовкам на классы (поле topic в датасете). Используйте 9 самых часто встречаемых топиков и 10-й для остальных, не вошедших в 9 классов. Оцените модель с помощью отчета о классификации, сделайте выводы.
# 3. На 5 баллов. Найдите публичный датасет по обращениям граждан в администрацию, техническую поддержку или за консультацией. Обучите модель классифицировать обращения по тематикам. Сформируйте отчет о классификации и матрицу ошибок.


# %%
# !wget https://storage.yandexcloud.net/academy.ai/Sarcasm_Headlines_Dataset_v2.json.zip
# !unzip -qo "Sarcasm_Headlines_Dataset_v2.json.zip" -d ./dataset

# 4. финальная установка auto-sklearn
# !pip install auto-sklearn
# !pip install autokeras==1.1.0 tensorflow==2.15.1 keras-nlp==0.5.1

# %%
# Библиотека матричного вычисления
# Библиотека для работы с регулярными выражениями
import re

# Библиотека AutoML autokeras
import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np

# Библиотека для работы с данными
import pandas as pd

# Библиотеки для построения графиков и их стилизации
import seaborn as sns

# Библиотека для работы с фреймворком TensorFlow
import tensorflow as tf

# Необходимые метрики для построения Матрицы ошибок и отчета о классификации
from sklearn.metrics import classification_report, confusion_matrix

# Утилита для расщепления выборки
from sklearn.model_selection import train_test_split

# %%
# %matplotlib inline

# %%
address = "./dataset/Sarcasm_Headlines_Dataset_v2.json"
json_df = pd.read_json(
    address, lines=True
)  # библиотека pandas умеет работать с json данными
df_sarcasm = pd.DataFrame(json_df)  # создаем датафрейм

df_sarcasm.head()  # выводим первые 5 записей датафрейма

# %%
df_sarcasm = df_sarcasm.drop("article_link", axis=1)

# %%
print("Найдено дубликатов: ", df_sarcasm.duplicated().sum())
# Удаляем дубликаты
df_sarcasm.drop_duplicates(subset=["headline"], inplace=True)
print("Осталось дубликатов после очистки: ", df_sarcasm.duplicated().sum())

# %%
X_train, X_tmp, y_train, y_tmp = train_test_split(
    np.array(df_sarcasm.headline), np.array(df_sarcasm.is_sarcastic), test_size=0.3
)

X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5)
