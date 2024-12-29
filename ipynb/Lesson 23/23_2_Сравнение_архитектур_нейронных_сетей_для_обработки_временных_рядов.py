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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/23_2_%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D1%85_%D1%81%D0%B5%D1%82%D0%B5%D0%B9_%D0%B4%D0%BB%D1%8F_%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B8_%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85_%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="ytrAthwP_atw"
# **Навигация по уроку**
#
# 1. [Анализ временных рядов](https://colab.research.google.com/drive/1q9nM-aWF6wZ2XuBxjQEAgjKxrL45axit)
# 2. Сравнение архитектур нейронных сетей для обработки временных рядов
# 3. [Домашняя работа](https://colab.research.google.com/drive/181g4qP5fB9PsqGCkcKAR37btxpRMU1Zt)

# %% [markdown] id="o4Um4hg1umc4"
# Мы продолжаем изучать временные ряды. Теперь нам необходимо изучить генератор временных рядов и применить его для подготовки данных и обучить на них различные архитектуры нейронных сетей. Сделать соответствующие выводы.

# %% [markdown] id="TISgmj29uwC2"
# ## Подготовка данных

# %% [markdown] id="JhCvuIE_vVBR"
# ### Импорт библиотек и загрузка данных

# %% id="ZApSR5aPgxnf"
# Работа с массивами
import numpy as np

# Работа с таблицами
import pandas as pd

# Построение моделей нейронных сетей
from keras.models import Sequential, Model

# Слои
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization
from keras.layers import Flatten, Conv1D, Conv2D, LSTM, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, RepeatVector

# Оптимизаторы
from keras.optimizers import Adam

# Генератор выборки временных рядов
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Нормировка
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Отрисовка графиков
import matplotlib.pyplot as plt
# %matplotlib inline

# Назначение размера и стиля графиков по умолчанию
from pylab import rcParams
plt.style.use('ggplot')
rcParams['figure.figsize'] = (14, 7)

# Отключение лишних предупреждений
import warnings
warnings.filterwarnings('ignore')

# %% colab={"base_uri": "https://localhost:8080/", "height": 422} id="zfJXOaRzvi7K" outputId="850ae45d-a618-4887-f14e-422e553b0b6e"
# импортируем файл с данными о котировках акций Apple
# !wget https://storage.yandexcloud.net/academy.ai/AAPL.csv

# Загрузим только необходимые колонки usecols, укажем, что колонку Date необходимо преобразовать в формат DateTime (parse_dates) и сделать индексом index_col
price = pd.read_csv("./AAPL.csv", index_col='Date', usecols = ['Adj Close', 'Volume', 'Date'], parse_dates=['Date'])
price.head()

# %% [markdown] id="wh2Eudgwytpf"
# ### Визуализация временного ряда

# %% [markdown] id="WZyyLGqf1jiO"
# Построим на одном полотне 2 графика с приведенной ценной закрытия (с учётом корпоративных действий) и объемом торгов по акции.

# %% colab={"base_uri": "https://localhost:8080/", "height": 707} id="j-dz8tB3zV-G" outputId="a4e54aba-bd8b-4ee0-f655-c22bad8a92f5"
# Задание полотна для графиков - два подграфика один под другим с общей осью x
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 13), sharex=True)


# Отрисовка одного канала данных
# От начальной точки start длиной length
ax1.plot(price.index, price['Adj Close'], label='Цена закрытия')

ax1.set_ylabel('Цена, руб')
ax1.legend()

# Канал volume (объем)
ax2.bar(x=price.index,
        height=price['Volume'],
        label='Объем')
ax2.set_ylabel('Сделки')
ax2.legend()

plt.xlabel('Время')
# Регулировка пределов оси x
# plt.xlim(0, length)
# Указание расположить подграфики плотнее друг к другу
plt.tight_layout()
# Фиксация графика
plt.show()

# %% [markdown] id="gzbqQUTr2AWs"
# Далее нам необходимо подготовить выборки.

# %% [markdown] id="9h3eJRhs2HLM"
# ### Генератор временных рядов

# %% [markdown] id="0FkNjLHT2R0h"
# Чтобы вручную не делить выборки для работы с временными рядами Keras предоставляет готовый инструмент **TimeseriesGenerator**.
#

# %% [markdown] id="Flylmd0y2o6x"
# Параметры генератора TimeseriesGenerator:
# * x_train – временной ряд, из которого собираются данные.
# * y_train – целевые значения.
# * length – длина выходных последовательностей, окно, которым генератор пройдется по данным.
# * sampling_rate – размер шага при выборке данных в x_train.
# * stride – указывает, на сколько элементов между двумя выборками осуществлять сдвиг.
# * batch_size – сколько элементов вернет генератор при обращении к нему.
#

# %% [markdown] id="l5kJe2JLBbt9"
# Чтобы понять как работает генератор, выполним простой пример:

# %% id="n8hGg8U2lPBR"

# %% colab={"base_uri": "https://localhost:8080/"} id="6NucFi9RJstw" outputId="7b3b97f6-e052-49ab-81fe-6d2af0852abf"
# Простая последовательность
series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# параметры генератора
n_input = 4
batch_size = 2
generator = TimeseriesGenerator(series, series, length=n_input, batch_size=batch_size)

# Число сгенерированных примеров
print('Число примеров: %d' % len(generator))

# Вывод для каждого примера
for i in range(len(generator)):
	x, y = generator[i]

	print('%s => %s' % (x, y))

# %% [markdown] id="s4IllCTeJtP-"
# Как видно из примера генератор в качестве входных данных выдает мини-батчи (в нашем случае с одним примером) с последовательностью из `n_input` значений, а в качестве выходного следующее значение в последовательности.
#
# Чтобы лучше понять как работают генераторы выполните 3-4 примера с разными значениями параметров генератора (изменяйте `n_input` и `batch_size`, перезапуская ячейку, проанализируйте вывод ячейки).

# %% [markdown] id="rQ71gMtxSen3"
# #### Подготовка данных для генератора

# %% [markdown] id="c1ccWbHfSsjo"
# **Шаг 1**. Для начала удалим столбец `Volume`, который нам был нужен для красивого графика. Обучение НС мы будем проводить только по приведенной цене закрытия `Adj Close`.

# %% id="BnER5ef0BhKf"
price.drop(columns=['Volume'], inplace=True)

# %% [markdown] id="_vgp4IsYTgh0"
# **Шаг 2**. Разобьем последовательность на тестовую и обучающую выборки. Дату, которую мы выбрали  - начало года, но пропустили период, который на фондовом рынке называется "гонка" или "ралли" (рост в предновогодние дни и после резкое падение) и период восстановления после "гонки", пока рынок отыграет свои позиции. Таким образом наша модель будет предсказывать поведение рынка после "гонки", что скорее всего приведет к смещению нашего графика с прогнозными ценами вверх или вниз (в зависимости от того, угадали ли мы, что рынок восстановился или нет).

# %% id="yEH7tvdG2pNw"
train_data = price[:'2023-01-10']
test_data = price['2023-01-20':]

# %% [markdown] id="s2tpU7qDwGRZ"
# Между выборками мы пропустили несколько дней, чтобы уменьшить влияние обучающей выборки на тестовую.

# %% [markdown] id="qj0NnIX9VqMH"
# **Шаг 3**. Не забываем про нормировку. Особенно если бы мы обучали одновременно модель на цене закрытия вместе с объемом торгов, так как эти данные разных порядков.

# %% [markdown] id="_q38CMQ7DBY5"
# При работе с временными рядами необходимо также нормирование данных перед их отправкой в генератор.
#
# Напомним, что **StandardScaler** приводит данные с параметрами среднее равное 0 и среднеквадратичным отклонением равное 1.
#
# **MinMaxScaler** может иметь два диапазона:  $[−1;1]$  и  $[0;1]$.
#
# Данные можно нормировать как все сразу, так и столбцы по отдельности.
#
# Для каждого нормирования важно подобрать правильную активационную функцию на выходном слое. Для **StandardScaler** используют `linear`, так как он выдает результат во всем числовом диапазоне. **Для MinMaxScaler** с диапазоном  $[−1;1]$, используют как `linear`, так и `tanh`. Для **MinMaxScaler** с диапазоном $[0;1]$  используют `sigmoid` и `relu`.

# %% id="rt3xx663k5nL" outputId="5d7f7dea-1852-4729-d176-327f28d55576" colab={"base_uri": "https://localhost:8080/", "height": 450}
train_data

# %% id="jhcLSSIQ_lsh"
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

# %% [markdown] id="zAr-elaQV96O"
# **Шаг 4**. Запускаем генератор для обучающей выборки. По 14 точкам примера, будем предсказывать следующую. Т.е. каждый обучающий пример будет содержать 14 входных точек и 1 целевую (или выходную).

# %% colab={"base_uri": "https://localhost:8080/"} id="SgXKVhGrzH0-" outputId="49ad3cff-8f9e-40f3-ee8d-b0b76cb6f1de"
# Проверка формы данных
print(f'Тренировочные данные: {scaled_train_data.shape}')
print(f'Тестовые данные: {scaled_test_data.shape}')

# %% [markdown] id="495MpiOizhK0"
# Мы будем обучать модель на малом числе образцов, так как берем дневные графики биржевых цен.

# %% colab={"base_uri": "https://localhost:8080/"} id="1Pznb0Yg_oea" outputId="ee0c86e9-7a0e-4e2d-fe53-54c294a249de"
n_input = 14  # Размерность входных данных
n_features = 1 # Размерность выходных данных
BATCH_SIZE = 1 # Размер пакета

generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=BATCH_SIZE)
print(f'Форма обучающего пакета: {generator[0][0].shape}, y: {generator[0][1].shape}')

# %% [markdown] id="acHIz7rmd77C"
# **Шаг 5**. Генерируем валидационную выборку из тестовых данных.

# %% colab={"base_uri": "https://localhost:8080/"} id="o583gskAcrQ-" outputId="5eab2481-1741-4e33-d13b-e85b467145b6"
validator = TimeseriesGenerator(scaled_test_data, scaled_test_data, length=n_input, batch_size=BATCH_SIZE)
print(f'Форма валидационного пакета: {validator[0][0].shape}, y: {validator[0][1].shape}')

# %% [markdown] id="PQH8hjpDeDog"
# **Шаг 6**. Генерируем тестовую выборку из тестовых данных, но один батч на всю выборку.

# %% colab={"base_uri": "https://localhost:8080/"} id="48r-5RN_eP8J" outputId="0044959f-3946-4391-98b7-9104cb8a02dc"
tester = TimeseriesGenerator(scaled_test_data, scaled_test_data, length=n_input, batch_size=scaled_test_data.shape[0])
x_test, y_test = tester[0]
print(f'Форма тестовой выборки: {x_test.shape}, y: {y_test.shape}')


# %% [markdown] id="wLjgufD1XW67"
# ## Вспомогательные функции

# %% [markdown] id="V-njYvvQXhfk"
# Определим вспомогательные функции, чтобы сделать код наших примеров более простым и лаконичным, и чтобы не отвлекал нас от главного.

# %% id="9HQl315Xnq-J"
# Объявление функции графика обучения
def history_plot(history, title):

    # Рисование графиков обучения
    fig = plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
    plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
    plt.title(f'{title}. График обучения')

    # Показываем только целые метки шкалы оси x
    fig.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Средняя ошибка')
    plt.legend()
    plt.show()

# Функция расчета корреляции для двух рядов
def correlate(a, b):
    return np.corrcoef(a, b)[0, 1]

# Функция визуализации результата предсказания сети и верных ответов

def show_predict(y_pred, y_true, title=''):
    fig = plt.figure(figsize=(14, 7))
    # Прогнозный ряд сдвигается на 1 шаг назад, так как предсказание делалось на 1 шаг вперед
    plt.plot(y_pred[1:], label=f'Прогноз')
    plt.plot(y_true[:-1], label=f'Базовый')
    plt.title(title)

    # Показываем только целые метки шкалы оси x
    fig.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('Дата (относительно начала выборки)')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()

# Функция расчета результата предсказания
def get_pred(model, # модель
             x_test, y_test, # тестовая выборка
             y_scaler): # масштабирующий объект для y

    # Вычисление и деномализация предсказания
    y_pred_unscaled = y_scaler.inverse_transform(model.predict(x_test, verbose=0))

    # Денормализация верных ответов
    y_test_unscaled = y_scaler.inverse_transform(y_test)

    # Возврат результата предсказания и верные ответы в исходном масштабе
    return y_pred_unscaled, y_test_unscaled



# Функция рисования корреляций прогнозного ряда и исходного со смещением
# break_step - ограничитель на число временных лагов

def show_corr(y_pred, y_true, title='', break_step=30):

    # выбираем наименьшее из длины y_len и break_step в качестве числа лагов для графика
    y_len = y_true.shape[0]
    steps = range(1, np.min([y_len+1, break_step+1]))

    # Вычисление коэффициентов корреляции базового ряда и предсказания с разным смещением
    cross_corr = [correlate(y_true[:-step, 0], y_pred[step:, 0]) for step in steps]

    # Вычисление коэффициентов автокорреляции базового ряда с разным смещением
    auto_corr = [correlate(y_true[:-step, 0], y_true[step:, 0]) for step in steps]

    plt.plot(steps, cross_corr, label=f'Прогноз')
    plt.plot(steps, auto_corr, label=f'Эталон')

    plt.title(title)

    # Назначение меток шкалы оси x
    plt.xticks(steps)
    plt.xlabel('Шаги смещения')
    plt.ylabel('Коэффициент корреляции')
    plt.legend()
    plt.show()


# %% [markdown] id="BfYENdPuXzZD"
# ## Сравнение архитектур

# %% [markdown] id="tVLRjGIXhpKr"
# ### 1. Полносвязная модель

# %% [markdown] id="5dtLRtoVhud4"
# #### Архитектура

# %% colab={"base_uri": "https://localhost:8080/", "height": 225} id="pMZW1wz1jMBz" outputId="39a8f3bc-d134-4b1b-fbf6-2b2e8037ec10"
# Простая полносвязная сеть
model_dense = Sequential()
model_dense.add(Dense(100, input_shape=generator[0][0].shape[1:], activation='relu'))
model_dense.add(Flatten())
model_dense.add(Dense(n_features, activation='linear'))

model_dense.compile(optimizer='adam', loss='mse')
model_dense.summary()

# %% [markdown] id="qvCmE_lAmffc"
# #### Обучение

# %% colab={"base_uri": "https://localhost:8080/"} id="CwGFCMpEl6Xf" outputId="9042edab-0ad0-4436-b81f-6c512f917f04"
model_dense.fit(generator, epochs=20, validation_data=validator)

# %% [markdown] id="mVBPI6-XhsXu"
# #### Графики

# %% colab={"base_uri": "https://localhost:8080/", "height": 647} id="jTyeYvR9mouJ" outputId="c4b115f6-3a9c-4841-c2de-f302227b2d03"
history_plot(model_dense.history, 'Полносвязная модель')

# %% colab={"base_uri": "https://localhost:8080/", "height": 647} id="Z320IzulqEn9" outputId="c0bf1318-c3d7-4373-d763-8abeb4831209"
# Получение денормализованного предсказания и данных базового ряда
y_pred, y_true = get_pred(model_dense, x_test, y_test, scaler)

# Отрисовка графика сопоставления базового и прогнозного рядов
show_predict(y_pred, y_true, title=f'Полносвязная модель. Сопоставление базового и прогнозного рядов')



# %% colab={"base_uri": "https://localhost:8080/", "height": 646} id="0Ai6gLkw11zl" outputId="7a4e802b-af86-4b08-c143-0243146e10d9"
# Отрисовка графика корреляционных коэффициентов до заданного максимума шагов смещения
show_corr(y_pred, y_true, title=f'Полносвязная модель. Корреляционные коэффициенты по шагам смещения')

# %% [markdown] id="-xMaYL7-X5nb"
# ### 2. Рекуррентная модель LSTM(50)

# %% [markdown] id="Xe1q4AutYR--"
# #### Архитектура

# %% colab={"base_uri": "https://localhost:8080/"} id="JT4E5hEE_uZD" outputId="f67ccfee-8a9f-49c6-f258-7d47b0b08e04"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=generator[0][0].shape[1:]))
lstm_model.add(Dense(10, activation='relu'))
lstm_model.add(Dense(n_features))

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

# %% [markdown] id="_8KK1EDNYaCA"
# #### Обучение

# %% colab={"base_uri": "https://localhost:8080/"} id="CyFWxMGE_xx2" outputId="a450da92-25de-4010-df3d-21e045e15834"
lstm_model.fit(generator, validation_data=validator, epochs=20)

# %% [markdown] id="VGUsc9TrYg2d"
# #### Графики

# %% colab={"base_uri": "https://localhost:8080/", "height": 647} id="ZR_32z7aav9V" outputId="b19e82c1-f8eb-4dbf-e76a-f66fdee64464"
history_plot(lstm_model.history, 'LSTM(50)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 647} id="RDyAWDDm0dFz" outputId="89c0eac9-4013-4eae-8099-a1900636c88a"
# Получение денормализованного предсказания и данных базового ряда
y_pred, y_true = get_pred(lstm_model, x_test, y_test, scaler)

# Отрисовка графика сопоставления базового и прогнозного рядов
show_predict(y_pred, y_true, title=f'LSTM(50) модель. Сопоставление базового и прогнозного рядов')

# %% colab={"base_uri": "https://localhost:8080/", "height": 646} id="y6jalVtIFJlH" outputId="060f1f26-e5b0-4494-faf6-8f6e5498db65"
# Отрисовка графика корреляционных коэффициентов до заданного максимума шагов смещения
show_corr(y_pred, y_true, title=f'LSTM(50) модель. Корреляционные коэффициенты по шагам смещения')

# %% [markdown] id="vIH0ZUvCFWOw"
# ### 3. Одномерная свертка Conv1D(64) х 2 + Dense(50)

# %% [markdown] id="jIDgHMeFFWO7"
# #### Архитектура

# %% colab={"base_uri": "https://localhost:8080/"} outputId="c838a0fd-7c8c-4dcd-ea1d-837caeaccd80" id="pI98MfqmFWO7"
# Модель с одномерной сверткой
model_conv = Sequential()
model_conv.add(Conv1D(64, 4, input_shape=generator[0][0].shape[1:], activation='relu'))
model_conv.add(Conv1D(64, 4, activation='relu'))
model_conv.add(MaxPooling1D())
model_conv.add(Flatten())
model_conv.add(Dense(50, activation='relu'))
model_conv.add(Dense(n_features, activation='linear'))

model_conv.compile(optimizer='adam', loss='mse')
model_conv.summary()

# %% [markdown] id="GEsy6wx2FWO7"
# #### Обучение

# %% colab={"base_uri": "https://localhost:8080/"} outputId="b157a3d3-a104-40c4-86bb-332d2e1adc98" id="dvQociJIFWO8"
model_conv.fit(generator, validation_data=validator, epochs=20)

# %% [markdown] id="QYokVhZ_FWO8"
# #### Графики

# %% colab={"base_uri": "https://localhost:8080/", "height": 647} id="93uKU4PlFWO8" outputId="ec8a5e81-3a14-4010-b74e-edae5018381a"
history_plot(model_conv.history, 'Conv1D(64) х 2 + Dense(50)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 647} id="EoKDWIN7FWO8" outputId="cc64074b-0df6-4148-d0cf-2b6b86263405"
# Получение денормализованного предсказания и данных базового ряда
y_pred, y_true = get_pred(model_conv, x_test, y_test, scaler)

# Отрисовка графика сопоставления базового и прогнозного рядов
show_predict(y_pred, y_true, title=f'Conv1D(64) х 2 + Dense(50). Сопоставление базового и прогнозного рядов')

# %% colab={"base_uri": "https://localhost:8080/", "height": 646} id="sFt960o_FWO8" outputId="a61212c1-ed6f-401c-8a5e-ad9189cc9d09"
# Отрисовка графика корреляционных коэффициентов до заданного максимума шагов смещения
show_corr(y_pred, y_true, title=f'Conv1D(64) х 2 + Dense(50). Корреляционные коэффициенты по шагам смещения')

# %% [markdown] id="1ln3HNFaFSM3"
# ### Выводы

# %% [markdown] id="ZO_MvnCALjI-"
# Мы рассмотрели 3 архитектуры нейронных сетей для анализа временных рядов и можно сделать следующие выводы:

# %% [markdown] id="3TsT50gGN5FV"
# * Мы видим, что все архитектуры имеют склонность к автокорреляции. Поэтому следует ожидать лучшие результаты при использовании обработки тренда.
# * Задача предсказания точной цены закрытия акций не всегда верна. Наиболее часто нейронную сеть используют для предсказания направления цены: будет ли цена расти или падать в следующем периоде.
# * В идеальном варианте график автокорреляции должен быть постоянно убывающим без холмика и стартовать максимально близко к 1. Такой идеальный случай нам показала модель LSTM. В двух других примерах прогнозный график корреляции стартует чуть ниже единицы, немного растет (или не убывает), и отодвигается от эталонной автокорреляции. Чем это плохо? При таком поведении графика прогнозной автокорреляции, часто можно заметить, что вместо предсказания действительного значения НС берет последнее значение из элемента выборки.
#
# * Лучшей моделью оказалась LSTM модель.
# * Худший результат показала одномерная свертка.
#
# * По графику автокорреляции можно предсказывать насколько эффективна наша модель. Если появляется холмик на графике корреляции в районе первых двух шагов, то мы имеем эффект автокорреляции. В идеале холмика быть не должно, а график предсказания должен быть как можно ближе к эталонному.
#
# > **ВАЖНО**. Даже если сеть достаточно точно предсказывает значение, оно не всегда может быть корректным. Нужно смотреть на график автокорреляции. Если график предсказания и реального значения имеют зависимость (наличие холмика, рост прогнозной автокорреляции на начальных шагах или большая удаленность между эталонной и прогнозной автокорреляции), то нейронка не предсказывает, а всего лишь повторяет то, что было несколько шагов назад.
#

# %% [markdown] id="XAxJuGmMXj8c"
# А впереди вас ждет [домашняя работа](https://colab.research.google.com/drive/181g4qP5fB9PsqGCkcKAR37btxpRMU1Zt).
