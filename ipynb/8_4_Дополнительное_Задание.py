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

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/8_3_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="8LIWrmOPjBO6"
# # Задача 1.
#
# Для выполнения задачи будет использован **csv файл “vgsale_1**”, содержащий данные о видеоиграх, выпущенных с 1980 по 2020 гг. Каждое из наблюдений в файле имеет 10 характеристик:
#
# - **Name** – название игры,
# - **Platform** – игровая платформа (PC, PSP, X360 и др.),
# - **Year** – год выпуска игры,
# - **Genre** – жанр игры,
# - **Publisher** – издатель игры,
# - **NA_Sales –** продажи в Северной Америке (в миллионах),
# - **EU_Sales** – продажи в Европе (в миллионах),
# - **JP_Sales** – продажи в Японии (в миллионах),
# - **Other_Sales** – продажи в остальных странах мира (в миллионах),
# - **Global_Sales** – объем продаж по всему миру.
#
#
# Загрузите файл **«vgsales_1.csv»** в объект **==DataFrame==**, рассчитайте необходимые показатели и визуализируйте информацию, используя функции любой библиотеки для визуализации данных. **Задание:**
#
# 1. **==Ответь==** на вопрос: игры каких жанров были наиболее популярны до 2000 года, а какие после?
# 2. **==Оцени==** популярность жанров по количеству выпущенных игр и по объему продаж по всему миру. Для визуализации полученных результатов **==используй==** столбчатые диаграммы.
#
#
# **Примечание.** Одна и та же игра может встречаться в выборке несколько раз, т.к. она может быть выпущена на нескольких платформах.
#
# 1. **==Отобрази==** на графике общее число видеоигр, выпущенных в каждом году.
# 2. **==Определи==** трех издателей, выпустивших наибольшее количество видеоигр. **==Изобрази==** количество выпущенных издателями видеоигр для каждой платформы на столбчатой диаграмме (можно использовать диаграмму с накоплением).
# 3. **==Отобрази==** на круговых диаграммах доли суммарного объема продаж с 1980 г. до 2000 г. и с 2000 г. до 2020 г. в Северной Америке, Европе, Японии (также для построения корректных диаграмм используйте столбец “Other Sales”) от объема продаж по всему миру.

# %% Cell
# %matplotlib inline

# %% Cell
# Загрузим все необходимые библиотеки
import ast
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from plotly.subplots import make_subplots

# %% Cell: load data
try:
    df = pd.read_csv("../assets/vgsale 1.csv")
except FileNotFoundError:
    df = pd.read_csv("vgsale 1.csv")

df = df.drop(df[df["Year"].isnull()].index)
df["Year"] = df["Year"].astype(int)

# %% Cell: calculate
# По количеству вышедших игр
genre_counts = df.groupby(["Year", "Genre"]).size().reset_index(name="Counts")
genre_counts["Total"] = genre_counts.groupby("Year")["Counts"].transform("sum")
genre_counts["Percentage"] = genre_counts["Counts"] / genre_counts["Total"] * 100

# %%
# По количеству проданных игр
genre_counts_sales = df.groupby(["Year", "Genre"])["Global_Sales"].sum().reset_index()
genre_counts_sales["Total"] = genre_counts.groupby("Year")["Global_Sales"].transform("sum")
genre_counts_sales["Percentage"] = genre_counts["Global_Sales"] / genre_counts["Total"] * 100

# %%
genre_counts.head()
# type(genre_counts)

# %%
_genre_counts = (
    genre_counts.groupby(["Year"] + ["Genre"])["Percentage"]
    .sum()
    .unstack(fill_value=0.0)
)

_genre_counts.head()
# _genre_counts.info()

# %%
sns.catplot(
    x="Genre",
    y="Percentage",
    hue="Year",
    data=genre_counts,
    kind="bar",
    height=6,
    aspect=2,
    hue_norm=(2000,),
)
plt.title("Популярность жанров на рынке за каждый год")
plt.show()

# %%
_genre_counts.plot(
    kind="bar",
    stacked="True",
    figsize=(15, 5),
    title="Популярность жанров на рынке за каждый год",
)

plt.legend(bbox_to_anchor=(1.01, 1.2), ncol=6)

plt.show()

# %%
# **==Отобрази==** на графике общее число видеоигр, выпущенных в каждом году.
_temp_df = df.groupby(["Year"])["Name"].count().reset_index().plot(kind="bar", x="Year")
plt.show()


# %%
# **==Определи==** трех издателей, выпустивших наибольшее количество видеоигр.
_temp_df = df.groupby(["Publisher"])["Name"].nunique().sort_values(ascending=False)
print(_temp_df.head(3))


# %% [markdown]
# #  **==Изобрази==** количество выпущенных издателями видеоигр для каждой платформы на столбчатой диаграмме (можно использовать диаграмму с накоплением).


# %% [markdown]
# # Задача 2.
#
# Для выполнения работы будет использован csv файл “IQ_countries”, содержащий данные о среднем значении IQ по странам мира. Каждое из наблюдений в файле имеет следующие характеристики:
#
# - **Rank** – место в рейтинге
# - **Country** – название страны
# - **Average IQ** – средний показатель IQ
# - **Continent –** название континента
# - **Literacy Rate** – коэффициент грамотности
# - **Nobel Prices** – количество нобелевских премий
# - **Human Development Index** – индекс человеческого развития
# - **Mean years of schooling** – среднее количество лет школы
# - **Gross National Income** – показатель “валовой национальный доход”
# - **Population** – численность населения.
#
# Задание: **==проведи==** разведочный анализ данных, **==выяви==** необычные взаимосвязи между значениями столбцов таблицы, **==выполни==** визуализацию, **==сделай==** выводы.
#
# Файл “IQ_countries.csv“ можно скачать по ссылке:

# %%

# %% [markdown]
# # Задача 3.
#
# Для выполнения работы будет использован csv файл “**shopping_habits**”, содержащий данные о различных покупках, которые совершаются покупателями в разных штатах США. Каждое из наблюдений в файле имеет следующие характеристики:
#
# - **Customer ID** – порядковый номер строки в таблице
# - **Age** – возраст покупателя
# - **Gender** – пол покупателя
# - **Item Purchased** – приобретенный товар
# - **Category** - категория
# - **Purchase Amount (USD)** – сумма покупки (в долларах)
# - **Location** – локация покупки
# - **Size** – размер (одежды)
# - **Color** – цвет
# - **Season** – время года совершения покупки
# - **Review Rating** – полученный в отзыве рейтинг
# - **Subscription Status** – статус подписки покупателя
# - **Shipping Type** – тип доставки
# - **Discount Applied** – применена ли скидка
# - **Promo Code Used** – применен ли промокод
# - **Previous Purchases –** были ли у данного покупателя предыдущие покупки
# - **Payment Method –** способ оплаты
# - **Frequency of Purchases** – частота покупок.
#
# Задание: ==**проведи**== разведочный анализ данных, **==выяви==** обычные взаимосвязи между значениями столбцов таблицы, **==выполни==** визуализацию, **==сделай==** выводы.
#
# Файл “shopping_habits.csv“ можно скачать по ссылке:

# %%
