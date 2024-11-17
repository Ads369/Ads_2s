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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/21_4_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="UDMQlLZbr8ra"
# **Навигация по уроку**
#
# 1. [Рекуррентные нейронные сети](https://colab.research.google.com/drive/1Mm5yFeJXZT9YcwlQMGx_T5JcEVgV8ZWy)
# 2. [Одномерные сверточные нейронные сети](https://colab.research.google.com/drive/1SCmcJdfsaxpJiQz_SOMH6gixV-43zPIB)
# 3. [Сравнение архитектур рекуррентных и одномерных сверточных сетей](https://colab.research.google.com/drive/15-SEqMwU3ALZmiEtlJFZllc38VTGHkGu)
# 4. Домашняя работа

# %% [markdown] id="sR5KZ9SVhfOY"
# Когда вы еще учились в школе, то вас часто мучали написанием сочинений. Может быть даже кто-нибудь из вас увлекался написанием стихотворений, романов или прозы. А значит, в вас живет дух великого русского писателя. А вот интересно какого? В данной домашней работе мы это и выясним!
#
# Чтобы узнать на какого писателя вы похожи необходимо выполнить следующее задание:
#
# 1. Скачать датасет с [писателями Русской литературы](https://storage.yandexcloud.net/academy.ai/russian_literature.zip). Каждый текст необходимо разбить на обучающую, проверочную и тестовую выборки, для этого модифицируйте функцию `seq_vectorize`, чтобы она возвращала все 3 выборки.
# 2. Используя материалы из ноутбука практического занятия [сравнение архитектур рекуррентных и одномерных сверточных сетей](https://colab.research.google.com/drive/15-SEqMwU3ALZmiEtlJFZllc38VTGHkGu), выберите лучший вариант нейронки и адаптируйте ее структуру.
# 3. Подгрузите веса Наташи как в уроке [20.3](https://colab.research.google.com/drive/1g_dX1XpRY--X6EjFflCC0717p9_9Y1SP) для слоя эмбендинга.
# 4. Заморозьте слой эмбединга.
# 5. Обучите модель на любом числе писателей (не менее 4-х) с балансировкой. Используйте обучающую выборку для обучения, а проверочную - в качестве валидационных данных (`validation_data`).
# 6. Постройте матрицу ошибок на тестовых образцах! В примерах мы строили на проверочных. Добейтесь средней точности более 70% на тестовых образцах. Получите 3 балла.
# 7. Если сможете добиться точности более 90% получите +1 балл.
# 8. Подготовьте свой текст и предложите нейронке предсказать на кого из русских писателей похож ваш текст. Вам необходимо построить круговую диаграмму с вероятностями предсказания моделью автора сочинения (по тестовой выборке). Если текст будет вашего авторства, то вы получите +1 балл. Если у вас плохо с фантазией, то возьмите небольшой фрагмент вашего любимого произведения любого автора, не представленного в датасете.

# %% cell
# !wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
# !pip install navec, wget
# %matplotlib inline


# %% cell
import glob  # Вспомогательный модуль для работы с файловой системой
import os  # Модуль для работы с файловой системой

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    MaxPooling1D,
    SimpleRNN,
    SpatialDropout1D,
)
from keras.models import Sequential
from keras.utils import get_file, to_categorical
from navec import Navec
from sklearn.metrics import (  # Для работы с матрицей ошибок
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from tensorflow.keras.preprocessing.text import Tokenizer

# %% cell
data_path = get_file(
    "russian_literature.zip",
    "https://storage.yandexcloud.net/academy.ai/russian_literature.zip",
)
# %% cell
import zipfile

zip_ref = zipfile.ZipFile(data_path, "r")
for file in zip_ref.namelist():
    try:
        zip_ref.extract(file, "./dataset")
    except OSError as e:
        if "[Errno 36] File name too long:" in str(e):
            new_filename = file[:50] + "." + file.split(".")[-1]
            with zip_ref.open(file) as source:
                with open(f"./dataset/{new_filename}", "wb") as target:
                    target.write(source.read())
import wget

url = "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar"
filename_navec = wget.download(url)
navec = Navec.load(filename_navec)

# %% cell
FILE_DIR_POEMS = "./dataset/poems"
file_list_poems = os.listdir(FILE_DIR_POEMS)
FILE_DIR_PROSE = "./dataset/prose"
file_list_prose = os.listdir(FILE_DIR_PROSE)
CLASS_LIST = list(set(file_list_poems + file_list_prose))
print("Общий список писателей:")
print(CLASS_LIST)
# %% cell
# Собираем в словарь весь датасет
all_texts = {}

for author in CLASS_LIST:
    all_texts[author] = ""
    for path in glob.glob("./dataset/prose/{}/*.txt".format(author)) + glob.glob(
        "./dataset/poems/{}/*.txt".format(author)
    ):
        with open(f"{path}", "r", errors="ignore") as f:
            text = f.read()
        all_texts[author] += " " + text.replace("\n", " ")

# %% cell
# Токенизация
embedding_dim = 300  # размерность векторов эмбединга (300d в имени эмбединга)
max_words = 15000  # Количество слов, рассматриваемых как признаки

# Используется встроенный в Keras токенизатор для разбиения текста и построения частотного словаря
tokenizer = Tokenizer(
    num_words=max_words,
    filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
    lower=True,
    split=" ",
    char_level=False,
)

# Построение частотного словаря по текстам
tokenizer.fit_on_texts(all_texts.values())

# %% cell
# Преобразуем текст в последовательности
seq_train = tokenizer.texts_to_sequences(all_texts.values())

# %% cell
# используем генератор цикла для получения длины текстов по каждому автору
total = sum(len(i) for i in seq_train)
print(f"Датасет состоит из {total} слов")

print("Общая выборка по писателям (по словам):")
mean_list = np.array([])
for author in CLASS_LIST:
    cls = CLASS_LIST.index(author)
    print(
        f"{author} - {len(seq_train[cls])} слов, доля в общей базе: {len(seq_train[cls])/total*100 :.2f}%"
    )
    mean_list = np.append(mean_list, len(seq_train[cls]))

print("Среднее значение слов: ", np.round(mean_list.mean()))
print("Медианное значение слов: ", np.median(mean_list))


median = int(np.median(mean_list))  # Зафиксировали медианное значение
CLASS_LIST_BALANCE = []  # Сбалансированный набор меток
seq_train_balance = []
for author in CLASS_LIST:
    cls = CLASS_LIST.index(author)
    if len(seq_train[cls]) > median * 0.6:
        seq_train_balance.append(seq_train[cls][:median])
        CLASS_LIST_BALANCE.append(author)


# %% cell
total = sum(len(i) for i in seq_train_balance)

print("Сбалансированная выборка по писателям (по словам):")
mean_list_balance = np.array([])
for author in CLASS_LIST_BALANCE:
    cls = CLASS_LIST_BALANCE.index(author)
    print(
        f"{author} - {len(seq_train_balance[cls])} слов, "
        f"доля в общей базе: {len(seq_train_balance[cls])/total*100 :.2f}%"
    )
    mean_list_balance = np.append(mean_list_balance, len(seq_train_balance[cls]))

print("Среднее значение слов: ", np.round(mean_list_balance.mean()))
print("Медианное значение слов: ", np.median(mean_list_balance))

# %% cell
fig, ax = plt.subplots()
ax.pie(
    [
        len(i) for i in seq_train_balance
    ],  # формируем список значений как длина символов текста каждого автора
    labels=CLASS_LIST_BALANCE,  # список меток
    pctdistance=1.2,  # дистанция размещения % (1 - граница окружности)
    labeldistance=1.4,  # размещение меток (1 - граница окружности)
    autopct="%1.2f%%",  # формат для % (2 знака после запятой)
)
plt.show()

# %% cell
# Нарезка примеров из текста методом скользящего окна
WIN_SIZE = 1000  # Ширина окна в токенах
WIN_STEP = 100  # Шаг окна в токенах


# Функция разбиения последовательности на отрезки скользящим окном
# Последовательность разбивается на части до последнего полного окна
# Параметры:
# sequence - последовательность токенов
# win_size - размер окна
# step - шаг окна
def seq_split(sequence, win_size, step):
    # Делим строку на отрезки с помощью генератора цикла
    return [
        sequence[i : i + win_size] for i in range(0, len(sequence) - win_size + 1, step)
    ]


def seq_vectorize(
    seq_list,  # Последовательность
    test_split,  # Доля на тестовую выборку
    class_list,  # Список классов
    win_size,  # Ширина скользящего окна
    step,  # Шаг скользящего окна
):
    # Списки для результирующих данных
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

    # Пробежимся по всем классам:
    for cls, class_item in enumerate(class_list):
        # Пороговое значение индекса для разбивки на тестовую и обучающую выборки
        gate_split = int(len(seq_list[cls]) * (1 - test_split))

        # Разбиваем последовательность токенов класса на отрезки
        vectors_train = seq_split(seq_list[cls][:gate_split], win_size, step)
        vectors_test_val = seq_split(seq_list[cls][gate_split:], win_size, step)

        # Разделение теста на val и test
        val_split = len(vectors_test_val) // 2
        vectors_val = vectors_test_val[:val_split]
        vectors_test = vectors_test_val[val_split:]

        # Добавляем отрезки в выборку
        x_train.extend(vectors_train)
        x_val.extend(vectors_val)
        x_test.extend(vectors_test)

        # Create one-hot encoded labels
        class_label = to_categorical(cls, len(class_list))

        # Для всех отрезков класса добавляем метки класса в виде one-hot-encoding
        # Каждую метку берем len(vectors) раз, так она одинакова для всех выборок одного класса
        y_train.extend([class_label] * len(vectors_train))
        y_val.extend([class_label] * len(vectors_val))
        y_test.extend([class_label] * len(vectors_test))

    # Возвращаем результатов как numpy-массивов
    return (
        np.array(x_train),
        np.array(y_train),
        np.array(x_val),
        np.array(y_val),
        np.array(x_test),
        np.array(y_test),
    )


# %% cell
x_train, y_train, x_val, y_val, x_test, y_test = seq_vectorize(
    seq_train_balance, 0.1, CLASS_LIST_BALANCE, WIN_SIZE, WIN_STEP
)
print(f"Форма входных данных для обучающей выборки: {x_train.shape}")
print(f"Форма выходных данных (меток) для обучающей выборки: {y_train.shape}")
print(f"Форма входных данных для валидационной выборки: {x_val.shape}")
print(f"Форма выходных данных (меток) для валидационной выборки: {y_val.shape}")
print(f"Форма входных данных для тестовой выборки: {x_test.shape}")
print(f"Форма выходных данных (меток) для тестовой выборки: {y_test.shape}")


# %% cell
# Шаг 8. Определим вспомагательные функции
# Вывод графиков точности и ошибки
def show_plot(history, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("График процесса обучения модели: " + title)
    ax1.plot(history.history["accuracy"], label="График точности на обучающей выборке")
    ax1.plot(
        history.history["val_accuracy"], label="График точности на проверочной выборке"
    )
    ax1.xaxis.get_major_locator().set_params(
        integer=True
    )  # На оси х показываем целые числа
    ax1.set_xlabel("Эпоха обучения")
    ax1.set_ylabel("График точности")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Ошибка на обучающей выборке")
    ax2.plot(history.history["val_loss"], label="Ошибка на проверочной выборке")
    ax2.xaxis.get_major_locator().set_params(
        integer=True
    )  # На оси х показываем целые числа
    ax2.set_xlabel("Эпоха обучения")
    ax2.set_ylabel("Ошибка")
    ax2.legend()
    plt.show()


# Функция вывода предсказанных значений
def show_confusion_matrix(y_true, y_pred, class_labels):
    # Матрица ошибок
    cm = confusion_matrix(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), normalize="true"
    )
    # Округление значений матрицы ошибок
    cm = np.around(cm, 3)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Матрица ошибок", fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()  # Убираем ненужную цветовую шкалу
    plt.xlabel("Предсказанные классы", fontsize=16)
    plt.ylabel("Верные классы", fontsize=16)
    fig.autofmt_xdate(rotation=45)  # Наклон меток горизонтальной оси
    plt.show()

    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    print(
        "\nСредняя точность распознавания: {:3.0f}%".format(
            100.0 * cm.diagonal().mean()
        )
    )


# %% cell
def loadEmbedding():
    word_index = tokenizer.word_index
    embeddings_index = navec

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


# %% Markdown
# # Рубрика Эксперименты!!!

# %% [markdown] id="Ik4cj2FtPXVw"
# #### **Embedding(Natasha) + SimpleRNN(10) + Dropout(0.3)**

# %% id="1FLZplc4QZ-b"
# Создание последовательной модели нейросети
model_SimpleRNN_1 = Sequential()

model_SimpleRNN_1.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)

# Слой регуляризации, "выключает" 1D карты объектов из эмбеддинг-векторов
model_SimpleRNN_1.add(SpatialDropout1D(0.3))
# Слой нормализации данных
model_SimpleRNN_1.add(BatchNormalization())
# Рекуррентный слой
model_SimpleRNN_1.add(SimpleRNN(10))
# Слой регуляризации Dropout для отдельных нейронов
model_SimpleRNN_1.add(Dropout(0.3))
# Выходной слой классификатора
model_SimpleRNN_1.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))

# %% id="LvPSibrmSWYk"
model_SimpleRNN_1 = loadEmbedding(model_SimpleRNN_1)

# %% colab={"base_uri": "https://localhost:8080/"} id="fTDGwqvqKTlw" outputId="6f23372b-4d60-4e0a-d743-32500d204e91"
model_SimpleRNN_1.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history = model_SimpleRNN_1.fit(
    x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val)
)


# %% colab={"base_uri": "https://localhost:8080/", "height": 400} id="tXhRPFU_IdIa" outputId="26e329a7-8593-4ca7-8be8-07a9afa13568"
show_plot(history, "Embedding(Natasha) + SimpleRNN(10) + Dropout(0.3)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="hXB8iTN2IdvE" outputId="3c1be2de-c3d2-4c97-d4ac-11deeb24ab96"
y_pred = model_SimpleRNN_1.predict(x_test)
show_confusion_matrix(y_test, y_pred, CLASS_LIST_BALANCE)

# %% [markdown] id="sTZGUmSiT6FO"
# #### **Embedding(Natasha) + SimpleRNN(5) + Dropout(0.2)**

# %% id="3AW4rqsuUVNR"
# Создание последовательной модели нейросети
model_SimpleRNN_2 = Sequential()

model_SimpleRNN_2.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)

# Слой регуляризации, "выключает" 1D карты объектов из эмбеддинг-векторов
model_SimpleRNN_2.add(SpatialDropout1D(0.2))
# Слой нормализации данных
model_SimpleRNN_2.add(BatchNormalization())
# Рекуррентный слой
model_SimpleRNN_2.add(SimpleRNN(5))
# Слой регуляризации Dropout для отдельных нейронов
model_SimpleRNN_2.add(Dropout(0.2))
# Выходной слой классификатора
model_SimpleRNN_2.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))

# %% id="53vi2wnDUgHg"
model_SimpleRNN_2 = loadEmbedding(model_SimpleRNN_2)

# %% colab={"base_uri": "https://localhost:8080/"} id="EVRFTgSRUyCX" outputId="faef14a9-6b7a-4282-81fa-fce91edc37df"
model_SimpleRNN_2.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history2 = model_SimpleRNN_2.fit(
    x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 293} id="cmjow47GUyh5" outputId="b7dd3244-e229-4053-d267-4ef1d0e0bf79"
show_plot(history2, "Embedding(Natasha) + SimpleRNN(40) + Dropout(0.3)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="HK3MjrZPVRiA" outputId="08da5de5-17a0-43e2-c947-09bd8f77fbf7"
y_pred2 = model_SimpleRNN_2.predict(x_test)
show_confusion_matrix(y_test, y_pred2, CLASS_LIST_BALANCE)

# %% [markdown] id="GxMHGdDWUu3D"
# #### **Embedding(Natasha) + GRU(10) + Dropout(0.2)**

# %% [markdown] id="BYbV58cXITpO"
# Слой GRU в Keras с параметрами регуляризации не считается на cuda ядрах GPU процессора. При запуске на GPU увидите предупреждение. Поэтому расчеты будут производиться долго и придется запастись терпением.

# %% id="gtCl2tu2YiHu"
model_GRU_3 = Sequential()
model_GRU_3.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)
model_GRU_3.add(SpatialDropout1D(0.2))
model_GRU_3.add(BatchNormalization())
# Рекуррентный слой GRU
model_GRU_3.add(GRU(10, dropout=0.2, recurrent_dropout=0.2, activation="relu"))
model_GRU_3.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))


# %% colab={"base_uri": "https://localhost:8080/"} id="YNPBsTd2Zf-J" outputId="6f22283e-259f-46c3-c83c-c9a6fe27d182"
model_GRU_3.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history3 = model_GRU_3.fit(
    x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 395} id="QdSrJMcfZg6-" outputId="4341062e-20ca-4f3d-c860-d997245f7c4e"
show_plot(history3, "Embedding(Natasha) + GRU(10) + Dropout(0.2)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="PuF07KGRZkDA" outputId="38a086a9-e98b-48ea-b35e-03941dd89c4b"
y_pred3 = model_GRU_3.predict(x_test)
show_confusion_matrix(y_test, y_pred3, CLASS_LIST_BALANCE)

# %% [markdown] id="VWYmu4eTbUIC"
# #### **Embedding(Natasha) + GRU(40) + Dropout(0.2)**

# %% id="UpglWSxebYMj"
model_GRU_4 = Sequential()
model_GRU_4.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)
model_GRU_4.add(SpatialDropout1D(0.2))
model_GRU_4.add(BatchNormalization())
# Рекуррентный слой GRU
model_GRU_4.add(GRU(40, dropout=0.2, recurrent_dropout=0.2, activation="relu"))
model_GRU_4.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))


# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 408} id="tinAn7GFbffT" outputId="d18aaafe-0dad-4ff8-d734-f9fbf9e341bf"
model_GRU_4.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history4 = model_GRU_4.fit(
    x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"background_save": true} id="kB6wWn4qbjSz" outputId="60d6b00a-a797-4410-ac30-7743f03bbe5a"
show_plot(history4, "Embedding(Natasha) + GRU(40) + Dropout(0.2)")

# %% colab={"background_save": true} id="sYKYzRxrbmsq" outputId="2ddaff8b-2464-456c-fba8-3a3978045d3a"
y_pred4 = model_GRU_4.predict(x_test)
show_confusion_matrix(y_test, y_pred4, CLASS_LIST_BALANCE)

# %% [markdown] id="YLH2PPQ0uG6m"
# #### **Embedding(Natasha) + LSTM(20) + Dropout(0.2)**

# %% id="PgrVnsL1uDNV"
model_LSTM_5 = Sequential()
model_LSTM_5.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)
model_LSTM_5.add(SpatialDropout1D(0.2))
model_LSTM_5.add(BatchNormalization())

# Рекуррентный слой LSTM
model_LSTM_5.add(LSTM(20))
model_LSTM_5.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))


# %% colab={"base_uri": "https://localhost:8080/"} id="PFu1Y1yGunR6" outputId="0e9f5ca6-99d8-4738-b94b-0ed35a755ac8"
model_LSTM_5.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history5 = model_LSTM_5.fit(
    x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 402} id="g-mzGAwYuq3N" outputId="74048bdc-8ee0-4329-d396-3c4aaa60ba25"
show_plot(history5, "Embedding(Natasha) + LSTM(20) + Dropout(0.2)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="DH1oxQg2uqyj" outputId="050694a3-f76e-42a9-c93e-98372992e7da"
y_pred5 = model_LSTM_5.predict(x_test)
show_confusion_matrix(y_test, y_pred5, CLASS_LIST_BALANCE)

# %% [markdown] id="rFamFFTAvPuT"
# #### **Embedding(Natasha) + LSTM(100) + Dropout(0.3)**

# %% id="weCdQaoBvPuU"
model_LSTM_6 = Sequential()
model_LSTM_6.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)
model_LSTM_6.add(SpatialDropout1D(0.3))
model_LSTM_6.add(BatchNormalization())

# Рекуррентный слой LSTM
model_LSTM_6.add(LSTM(100))
model_LSTM_6.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))


# %% colab={"base_uri": "https://localhost:8080/"} id="lQ9pgMbwvPuU" outputId="da070a06-3fb0-4ad7-b528-681c75b5185f"
model_LSTM_6.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history6 = model_LSTM_6.fit(
    x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 497} id="Ii4jOw3wvPuU" outputId="11688f01-2a73-4237-faa1-fa9ac16336d3"
show_plot(history6, "Embedding(Natasha) + LSTM(100) + Dropout(0.3)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="9lg2eY0LvPuU" outputId="c5cc78d1-c49b-4341-95ca-cf97048c7769"
y_pred6 = model_LSTM_6.predict(x_test)
show_confusion_matrix(y_test, y_pred6, CLASS_LIST_BALANCE)

# %% [markdown] id="c5RfIC5loJ_E"
# #### **Embedding(Natasha) + BLSTM(8)x2 + GRU(16)x2 + Dropout(0.3) + Dense(100)**

# %% [markdown] id="7dwIyZ-nzYoY"
# `Bidirectional(LSTM(8, return_sequences=True))` – этот слой
# активизирует двунаправленную сеть **LSTM**;
#
# `GRU(16, return_sequences=True, reset_after=True)` – параметр `reset_after=True` означает сброс данных.

# %% id="H6jCCMhPkseq"
model_MIX = Sequential()
model_MIX.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)
model_MIX.add(SpatialDropout1D(0.3))
model_MIX.add(BatchNormalization())

# Два двунаправленных рекуррентных слоя LSTM
model_MIX.add(Bidirectional(LSTM(8, return_sequences=True)))
model_MIX.add(Bidirectional(LSTM(8, return_sequences=True)))
model_MIX.add(Dropout(0.3))
model_MIX.add(BatchNormalization())

# Два рекуррентных слоя GRU
model_MIX.add(GRU(16, return_sequences=True, reset_after=True))
model_MIX.add(GRU(16, reset_after=True))
model_MIX.add(Dropout(0.3))
model_MIX.add(BatchNormalization())

# Дополнительный полносвязный слой
model_MIX.add(Dense(100, activation="relu"))
model_MIX.add(Dropout(0.3))
model_MIX.add(BatchNormalization())
model_MIX.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))


# %% colab={"base_uri": "https://localhost:8080/"} id="m5kGicpop4BX" outputId="6167c480-124c-42a7-908a-4ef8cc419f5d"
model_MIX.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history_mix = model_MIX.fit(
    x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 402} id="HlsaGYsVp6aw" outputId="a01026fa-9ee3-41ed-9490-ac8ad97da4ab"
show_plot(
    history_mix,
    "Embedding(Natasha) + BLSTM(8)x2 + GRU(16)x2 + Dropout(0.3) + Dense(100)",
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="8X9z3bSWp81A" outputId="851f3d68-90dd-4622-d64f-e51af396006e"
y_pred_mix = model_MIX.predict(x_test)
show_confusion_matrix(y_test, y_pred_mix, CLASS_LIST_BALANCE)

# %% [markdown] id="jXKOhmD35VZb"
# #### **Embedding(Natasha) + Conv1D(20)x2 + Dropout(0.2)**

# %% id="MfXu7yDh5Vy6"
model_Conv1D = Sequential()
model_Conv1D.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)
model_Conv1D.add(SpatialDropout1D(0.2))
model_Conv1D.add(BatchNormalization())
# Два слоя одномерной свертки Conv1D
model_Conv1D.add(Conv1D(20, 5, activation="relu", padding="same"))
model_Conv1D.add(Conv1D(20, 5, activation="relu"))
# Слой подвыборки/пулинга с функцией максимума
model_Conv1D.add(MaxPooling1D(2))
model_Conv1D.add(Dropout(0.2))
# Слой пакетной нормализации
model_Conv1D.add(BatchNormalization())
# Слой выравнивания в вектор
model_Conv1D.add(Flatten())
model_Conv1D.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))


# %% colab={"base_uri": "https://localhost:8080/"} id="235x0NVq7MfZ" outputId="ad34b251-8e6b-461d-c1b9-cadaa46c7738"
model_Conv1D.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history_1D = model_Conv1D.fit(
    x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 287} id="8Ma02eMM7RUa" outputId="e9cb6543-82d0-4004-e4c4-8deeee769770"
show_plot(history_1D, "Embedding(Natasha) + Conv1D(20)x2 + Dropout(0.2)")

# %% colab={"base_uri": "https://localhost:8080/", "height": 877} id="q81n6rBd7gQr" outputId="3e95728c-edad-4f75-f4cb-0969d07a057a"
y_pred_1d = model_Conv1D.predict(x_test)
show_confusion_matrix(y_test, y_pred_1d, CLASS_LIST_BALANCE)

# %% [markdown] id="7riJKlyc5mO3"
# #### **Embedding(Natasha) + Conv1D(20)x3 + LSTM(4)x2 + Dropout(0.2) + Dense(100)**

# %% id="YzYoqAol5mun"
model_Conv_LSTM = Sequential()
model_Conv_LSTM.add(
    Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=loadEmbedding())
)
model_Conv_LSTM.add(SpatialDropout1D(0.2))
# Рекуррентный слой LSTM
model_Conv_LSTM.add(LSTM(4, return_sequences=1))
# Полносвязный слой
model_Conv_LSTM.add(Dense(100, activation="relu"))
# Сверточный слой
model_Conv_LSTM.add(Conv1D(20, 5, activation="relu"))
# Рекуррентный слой LSTM
model_Conv_LSTM.add(LSTM(4, return_sequences=1))
# Слой регуляризации Dropout
model_Conv_LSTM.add(Dropout(0.2))
# Слой пакетной нормализации
model_Conv_LSTM.add(BatchNormalization())
# Два сверточных слоя
model_Conv_LSTM.add(Conv1D(20, 5, activation="relu"))
model_Conv_LSTM.add(Conv1D(20, 5, activation="relu"))
# Слой подвыборки/пулинга с функцией максимума
model_Conv_LSTM.add(MaxPooling1D(2))
model_Conv_LSTM.add(Dropout(0.2))
model_Conv_LSTM.add(BatchNormalization())
# Слой выравнивания в вектор
model_Conv_LSTM.add(Flatten())
model_Conv_LSTM.add(Dense(len(CLASS_LIST_BALANCE), activation="softmax"))


# %% colab={"base_uri": "https://localhost:8080/"} id="1ZKryvBN79W9" outputId="138d01f2-df18-4e1a-c175-b2b42a62fc05"
model_Conv_LSTM.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
history_1D_LSTM = model_Conv_LSTM.fit(
    x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 287} id="eysNPJan79W9" outputId="88f864d4-dc37-45ca-b51e-2ab15ca5e4d8"
show_plot(
    history_1D_LSTM,
    "Embedding(Natasha) + Conv1D(20)x3 + LSTM(4)x2 + Dropout(0.2) + Dense(100)",
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 877} id="sHT3Sm9b79W-" outputId="84f4f6f2-15ae-4000-9778-1000aea3df00"
y_pred_lstm = model_Conv_LSTM.predict(x_test)
show_confusion_matrix(y_test, y_pred_lstm, CLASS_LIST_BALANCE)


# %% [markdown]
# 1. Embedding(Natasha) + SimpleRNN(10) + Dropout(0.3) - **19%**
# 2. Embedding(Natasha) + SimpleRNN(5) + Dropout(0.2)  - **20%**
# 3. Embedding(Natasha) + GRU(10) + Dropout(0.2) - **75%**
# 4. Embedding(Natasha) + GRU(40) + Dropout(0.2) - **86%**
# 5. Embedding(Natasha) + LSTM(20) + Dropout(0.2) - **83%**
# 6. Embedding(Natasha) + LSTM(100) + Dropout(0.3) - **83%**
# 7. Embedding(Natasha) + BLSTM(8)x2 + GRU(16)x2 + Dropout(0.3) + Dense(100) - **89%**
# 8. Embedding(Natasha) + Conv1D(20)x2 + Dropout(0.2) - **76%**
# 9. Embedding(Natasha) + Conv1D(20)x3 + LSTM(4)x2 + Dropout(0.2) + Dense(100) - **77%**
