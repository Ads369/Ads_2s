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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/23_3_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="fEX_t6f7_cdz"
# **Навигация по уроку**
#
# 1. [Анализ временных рядов с помощью НС](https://colab.research.google.com/drive/1q9nM-aWF6wZ2XuBxjQEAgjKxrL45axit)
# 2. [Сравнение архитектур нейронных сетей для обработки временных рядов](https://colab.research.google.com/drive/1-D-qXFYJ9b5sLLz_CFkUYmR-I2tM7KO2)
# 3. Домашняя работа

# %% [markdown] id="eMEgCDqiqDvi"
# **В домашней работе вам необходимо:**
# 1. Выбрать любую понравившуюся модель из [практической](https://colab.research.google.com/drive/1-D-qXFYJ9b5sLLz_CFkUYmR-I2tM7KO2) части урока.
# 2. Используя известный [датасет](https://storage.yandexcloud.net/academy.ai/AAPL.csv) котировок Apple, обучить модель. Вывести графики из урока: график процесса обучения, сопоставления базового и прогнозного рядов, а также график автокорреляции.
# 3. Для получения трех проходных баллов за урок необходимо скорректировать код урока для данных с batch_size не равному 1.
# 4. Хотите 4 балла? Возьмите полносвязанную модель или с одномерной сверткой. Добейтесь подбором параметров и выбором архитектуры идеального графика автокорреляции без холмиков, равномерно спадающих графиков эталонной и прогнозной автокорреляции, максимально близко друг к другу.
# 5. Для получения дополнительного балла вам необходимо избавиться от тренда с помощью дифференцирования в датасете.
# 6. Еще один балл можно получить сверху, если догадаетесь как на графике сопоставления базового и прогнозного рядов отобразить реальную дату, а не относительную.


# %% id="ZApSR5aPgxnf"
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import (
    LSTM,
    Dense,
)
from keras.models import Sequential
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

plt.style.use("ggplot")
rcParams["figure.figsize"] = (14, 7)
warnings.filterwarnings("ignore")

# %% cell
# %matplotlib inline
# !wget https://storage.yandexcloud.net/academy.ai/AAPL.csv


# %% cell
price = pd.read_csv(
    "./AAPL.csv",
    index_col="Date",
    usecols=["Adj Close", "Volume", "Date"],
    parse_dates=["Date"],
)

# Fix trand
price["diff"] = price["Adj Close"].diff()
price = price.dropna()

price.head()

# %%
# Задание полотна для графиков - два подграфика один под другим с общей осью x
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 13), sharex=True)


ax1.plot(price.index, price["Adj Close"], label="Цена закрытия")

ax1.set_ylabel("Цена, руб")
ax1.legend()

# Канал volume (объем)
ax2.bar(x=price.index, height=price["Volume"], label="Объем")
ax2.set_ylabel("Сделки")
ax2.legend()

plt.xlabel("Время")
# Указание расположить подграфики плотнее друг к другу
plt.tight_layout()
# Фиксация графика
plt.show()

# %%
price.drop(columns=["Volume"], inplace=True)
price.drop(columns=["Adj Close"], inplace=True)
price.head()

# %%
# Split the data
test_size = round(len(price) * 0.8)
train_data = price[:test_size]
test_data = price[test_size:]

validate_size = len(test_data) // 2
validate_data = test_data[:validate_size]
test_data = test_data[validate_size:]

print(f"Тренировочные данные: {train_data.shape}")
print(f"Тестовые данные: {test_data.shape}")
print(f"Валидационные данные: {validate_data.shape}")


# %%
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
scaled_validate_data = scaler.transform(validate_data)

print(f"Тренировочные данные: {scaled_train_data.shape}")
print(f"Тестовые данные: {scaled_test_data.shape}")
print(f"Валидационные данные: {scaled_validate_data.shape}")


# %%
n_input = 14  # Размерность входных данных
n_features = 1  # Размерность выходных данных
# BATCH_SIZE = 1  # Размер пакета
BATCH_SIZE = 4

generator = TimeseriesGenerator(
    scaled_train_data, scaled_train_data, length=n_input, batch_size=BATCH_SIZE
)
print(f"Форма обучающего пакета: {generator[0][0].shape}, y: {generator[0][1].shape}")

validator = TimeseriesGenerator(
    scaled_validate_data, scaled_validate_data, length=n_input, batch_size=BATCH_SIZE
)
print(
    f"Форма валидационного пакета: {validator[0][0].shape}, y: {validator[0][1].shape}"
)

tester = TimeseriesGenerator(
    scaled_test_data,
    scaled_test_data,
    length=n_input,
    batch_size=scaled_test_data.shape[0],
    # batch_size=BATCH_SIZE,
)
x_test, y_test = tester[0]
print(f"Форма тестовой выборки: {x_test.shape}, y: {y_test.shape}")

# %%
# ## Вспомогательные функции


# Объявление функции графика обучения
def history_plot(history, title):
    # Рисование графиков обучения
    fig = plt.figure(figsize=(14, 7))
    plt.plot(history.history["loss"], label="Ошибка на обучающем наборе")
    plt.plot(history.history["val_loss"], label="Ошибка на проверочном наборе")
    plt.title(f"{title}. График обучения")

    # Показываем только целые метки шкалы оси x
    fig.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel("Эпоха обучения")
    plt.ylabel("Средняя ошибка")
    plt.legend()
    plt.show()


# Функция расчета корреляции для двух рядов
def correlate(a, b):
    return np.corrcoef(a, b)[0, 1]


# Функция визуализации результата предсказания сети и верных ответов
def show_predict(y_pred, y_true, dates, title=""):
    plot_dates = dates[-1 * len(y_pred) + 1 :]

    # Plot with dates on x-axis
    plt.plot(plot_dates, y_pred[1:], label="Прогноз")
    plt.plot(plot_dates, y_true[:-1], label="Базовый")
    plt.title(title)
    plt.xlabel("Дата (относительно начала выборки)")
    plt.ylabel("Значение")
    plt.legend()
    plt.show()


# Функция расчета результата предсказания
def get_pred(
    model,  # модель
    x_test,
    y_test,  # тестовая выборка
    y_scaler,  # масштабирующий объект для y
):
    n_samples = x_test.shape[0]
    if n_samples % BATCH_SIZE != 0:
        pad_size = BATCH_SIZE - (n_samples % BATCH_SIZE)
        x_test_padded = np.pad(x_test, ((0, pad_size), (0, 0), (0, 0)), mode="edge")
        y_pred = model.predict(x_test_padded, batch_size=BATCH_SIZE, verbose=0)
    else:
        y_pred = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)

    y_pred_unscaled = y_scaler.inverse_transform(y_pred)
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    return y_pred_unscaled, y_test_unscaled


# Функция рисования корреляций прогнозного ряда и исходного со смещением
# break_step - ограничитель на число временных лагов
def show_corr(y_pred, y_true, title="", break_step=30):
    # выбираем наименьшее из длины y_len и break_step в качестве числа лагов для графика
    y_len = y_true.shape[0]
    steps = range(1, np.min([y_len + 1, break_step + 1]))

    # Вычисление коэффициентов корреляции базового ряда и предсказания с разным смещением
    cross_corr = [correlate(y_true[:-step, 0], y_pred[step:, 0]) for step in steps]

    # Вычисление коэффициентов автокорреляции базового ряда с разным смещением
    auto_corr = [correlate(y_true[:-step, 0], y_true[step:, 0]) for step in steps]

    plt.plot(steps, cross_corr, label="Прогноз")
    plt.plot(steps, auto_corr, label="Эталон")

    plt.title(title)

    # Назначение меток шкалы оси x
    plt.xticks(steps)
    plt.xlabel("Шаги смещения")
    plt.ylabel("Коэффициент корреляции")
    plt.legend()
    plt.show()


# %%
# ### 2. Рекуррентная модель LSTM(50)

lstm_model = Sequential()
# lstm_model.add(LSTM(50, activation="relu", input_shape=generator[0][0].shape[1:]))
lstm_model.add(
    LSTM(
        50,
        activation="relu",
        input_shape=(n_input, n_features),  # Explicitly specify input shape
    )
)  # Add batch_input_shape
lstm_model.add(Dense(10, activation="relu"))
lstm_model.add(Dense(n_features))

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.summary()


# %%
steps_per_epoch = len(scaled_train_data) // BATCH_SIZE
validation_steps = len(scaled_validate_data) // BATCH_SIZE

lstm_model.fit(
    generator,
    validation_data=validator,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

# %%
# #### Графики
history_plot(lstm_model.history, "LSTM(50)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 647} id="RDyAWDDm0dFz" outputId="89c0eac9-4013-4eae-8099-a1900636c88a"
test_dates = test_data.index
# Получение денормализованного предсказания и данных базового ряда
y_pred, y_true = get_pred(lstm_model, x_test, y_test, scaler)

# Отрисовка графика сопоставления базового и прогнозного рядов
show_predict(
    y_pred,
    y_true,
    test_dates,
    title="LSTM(50) модель. Сопоставление базового и прогнозного рядов",
)

# %%
# Отрисовка графика корреляционных коэффициентов до заданного максимума шагов смещения
show_corr(
    y_pred,
    y_true,
    title="LSTM(50) модель. Корреляционные коэффициенты по шагам смещения",
)

# %%
# импортируем функцию seasonal_decompose из statsmodels
# задаем размер графика
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams["figure.figsize"] = 11, 9

# qwe = np.diff(price['Adj Close'])
# qwe = np.diff(price['diff'])

# применяем функцию к данным о котировках
decompose = seasonal_decompose(price.resample(rule="1M").mean())
decompose.plot()

plt.show()


# %%
def get_corr_coef(
    data,  # данные
    lag,  # временной лаг
):
    array_data = np.asarray(data)  # Преобразуем в массив NumPy

    # вычисление матрицы коэффициентов и отбор коэффициента из нужного места матрицы
    return np.corrcoef(array_data[lag:], array_data[:-lag])[0, 1]


# %%
np.round(get_corr_coef(price["diff"], 5), 2)
# импортируем автокорреляционную функцию (ACF)
from statsmodels.graphics.tsaplots import plot_acf

# применим функцию к нашему набору данных
# параметр lags - сколько мы хотим отобразить лагов
plot_acf(price["diff"], lags=100)

# добавим отступы сверху и снизу на графике
plt.axis("tight")
plt.title("Влияние тренда на автокорреляцию котировок акций")
plt.xlabel("Временной лаг", fontsize=16)
plt.ylabel("Коэффициент корреляции", fontsize=16)
plt.show()
