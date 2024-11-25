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

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/22_3_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="Wscc0Fl5qhuR"
#  **Навигация по уроку**
#
# 1. [Решение задач регрессии с помощью НС](https://colab.research.google.com/drive/1GbXbqPbC4A2NVJEj-5pOOySN3gyalBQN)
# 2. [Анализ резюме кандидатов](https://colab.research.google.com/drive/1L4pI4giYvWY3T4gfqTF5c_XFkN9k4Nfz)
# 3. Домашняя работа

# %% [markdown] id="_up7JAoo8CF0"
# **В домашней работе необходимо выполнить следующее задание:**
#
# 1. Используя предложенный [датасет](https://storage.yandexcloud.net/academy.ai/japan_cars_dataset.csv) японских машин, обучите модель предсказывать цены на японские автомобили.
# 2. Создайте обучающую, тестовую и проверочную выборки.
# 3. Оцените качество работы созданной сети, определите средний процент ошибки на проверочной выборке.
# 4. В качестве ошибки рекомендуется использовать среднеквадратическую ошибку (mse).
# 5. Выполнив задание, получите 3 балла.
# 6. Хотите 4 балла? Добейтесь ошибки менее 10%.
# 7. Хотите 5 баллов? Добейтесь ошибки менее 5%.
#

# %% [markdown] id="7b9zW5YN6znk"
# **Примечание**. Подробную информацию о датасете можно узнать на портале соревновани [kaggle.com](https://www.kaggle.com/datasets/doaaalsenani/used-cars-dataets/data).
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="BDCOnE5A7XiG" outputId="c6c455d1-0d56-4c8d-cdfd-a281ae0c0836"
# !wget https://storage.yandexcloud.net/academy.ai/japan_cars_dataset.csv

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="NS_vFnv17mjE" outputId="99be074f-2971-4b03-855f-cbfc855c6347"
import pandas as pd
cars = pd.read_csv('japan_cars_dataset.csv', sep=',')

# Удалим строки с пустыми значениями
cars = cars.dropna()

# Выводим первые 10 машин
cars.head(10)

# %% id="b-a8LLHThFg8"
# ваше решение
