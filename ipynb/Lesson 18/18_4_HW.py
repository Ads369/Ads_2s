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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/18_4_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="6rS32nxCMMVV"
# **Навигация по уроку**
#
# 1. [Введение в сверточные нейронные сети](https://colab.research.google.com/drive/10cBnEfHhZlv3ZhEgVimA3GqSEZtvqpTv)
# 2. [Обучение сверточной нейронной сети на ограниченном наборе данных](https://colab.research.google.com/drive/1e1aZ9K1vQIujPf1mzmta5xtbfNqJ_ai-)
# 3. [Предобученные сверточные НС](https://colab.research.google.com/drive/12VehrJe062P9QImtvjILQG0DOLooiyk0)
# 4. Домашняя работа

# %% [markdown] id="zd9JxveEPUco"
# В данном домашнем задании вам необходимо:
#
# 1. Используйте датасет "Собаки и кошки", рассмотренный в данном уроке. Причем используйте его целиком, а не только 4000 изображений.
# 2. Проведите аугментацию изображений.
# 3. В качестве предобученной модели возьмите `MobileNet`
# 4. Создайте модель, приведенную ниже.
# 5. Обучите модель и проверьте на тестовой выборке.
# 6. Если модель не обеспечивает заданную точность - "поиграйтесь" с гиперпараметрами.
#
#
# Для получения 3 баллов за задание необходимо достичь на контрольной выборке точности 90%, 4 баллов -  более 93%, 5 баллов - более 95%.
#
# На 20 тыс. изображений данная модель выдавала нам результат 99%.
#
# **Подсказка**. Обратите внимание, что предлагаемая модель уже не является бинарной классификацией. Это уже задача многоклассовой классификации (в нашем случае 2 класса). А значит в генераторах изображений необходимо использовать:
#
# ```pyton
# def model_maker():
#     base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#
#     for layer in base_model.layers[:]:
#         layer.trainable = False
#
#     input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#     custom_model = base_model(input)
#     custom_model = GlobalAveragePooling2D()(custom_model)
#     custom_model = Dense(64, activation="relu")(custom_model)
#     custom_model = Dropout(0.5)(custom_model)
#     predictions = Dense(NUM_CLASSES, activation="softmax")(custom_model)
#
#     return Model(inputs=input, outputs=predictions)
#
# ```
#
# ```pyton
# class_mode='categorical'
# ```
#
# Также необходимо вспомнить какую функцию ошибки использовать с задачей многоклассовой классификации.
# Можно попробовать в качестве оптимизатора использовать Adam с разными шагами.
#
# Также обратите внимание, что вместо слоя `Flatten()`, вам предлагается использовать `GlobalAveragePooling2D()` (https://keras.io/api/layers/pooling_layers/global_average_pooling2d/).
#


# %%
# # # @title Загрузка набора данных и обучение модели
# !wget https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip
# !unzip -qo "cat-and-dog" -d ./temp


# %% Import
import os
import shutil  # Набор утилит для работы с файловой системой  # Набор утилит для работы с файловой системой
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras import (
    Model,
    layers,  # импортируем слои  # импортируем слои
    models,  # импортируем модели  # импортируем модели
    optimizers,  # импортируем функции оптимизации
)
from keras.applications import VGG16, MobileNet
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% cell
# Папка с папками картинок, рассортированных по категориям
IMAGE_PATH = "./temp/training_set/training_set/"
IMAGE_PATH_TEST = "./temp/test_set/test_set/"
IMG_WIDTH = 150
IMG_HEIGHT = 150

# Папка в которой будем создавать выборки
DATASET_DIR = "./dataset/"
CLASS_LIST = sorted(os.listdir(IMAGE_PATH))
CLASS_COUNT = len(CLASS_LIST)

# При повторном запуске пересоздаим структуру каталогов
# Если папка существует, то удаляем ее со всеми вложенными каталогами и файлами
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

# Создаем папку по пути BASE_DIR
os.mkdir(DATASET_DIR)

# Сцепляем путь до папки с именем вложенной папки. Аналогично BASE_DIR + '/train'
train_dir = os.path.join(DATASET_DIR, "train")
validation_dir = os.path.join(DATASET_DIR, "validation")
test_dir = os.path.join(DATASET_DIR, "test")

os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)


data_files = []  # Cписок путей к файлам изображений
data_labels = []  # Список меток классов

# %% cell
for class_label in range(CLASS_COUNT):
    class_name = CLASS_LIST[class_label]  # Выборка имени класса из списка имен
    class_path = IMAGE_PATH + class_name  # Полный путь к папке с изображениями класса

    # Получение списка имен файлов с изображениями текущего класса
    class_files = os.listdir(class_path)

    # Вывод информации о численности класса
    print(f"Размер класса {class_name} составляет {len(class_files)} животных")

    # Добавление к общему списку всех файлов класса с добавлением родительского пути
    data_files += [f"{class_path}/{file_name}" for file_name in class_files]

    # Добавление к общему списку меток текущего класса - их ровно столько, сколько файлов в классе
    data_labels += [class_label] * len(class_files)

print("Общий размер базы для обучения:", len(data_labels))

# %% cell

# Функция создания подвыборок (папок с файлами)
def create_dataset(
    img_path: str,  # Путь к файлам с изображениями классов
    new_path: str,  # Путь к папке с выборками
    class_name: str,  # Имя класса (оно же и имя папки)
    start_index: int = 0,  # Стартовый индекс изображения, с которого начинаем подвыборку
    end_index: int = -1,  # Конечный индекс изображения, до которого создаем подвыборку
):
    src_path = os.path.join(img_path, class_name)
    dst_path = os.path.join(new_path, class_name)

    # Получение списка имен файлов с изображениями текущего класса
    class_files = os.listdir(src_path)

    # Создаем подпапку, используя путь
    os.mkdir(dst_path)

    # Перебираем элементы, отобранного списка с начального по конечный индекс
    for fname in class_files[start_index:end_index]:
        src = os.path.join(src_path, fname)
        dst = os.path.join(dst_path, fname)
        # Копируем файл из источника в новое место (назначение)
        shutil.copyfile(src, dst)


for class_label in range(
    CLASS_COUNT
):  # Перебор по всем классам по порядку номеров (их меток)
    class_name = CLASS_LIST[class_label]  # Выборка имени класса из списка имен

    create_dataset(IMAGE_PATH, train_dir, class_name, 0)
    create_dataset(IMAGE_PATH, validation_dir, class_name, 0)
    create_dataset(IMAGE_PATH_TEST, test_dir, class_name, 0)


# %% cell
# conv_base = VGG16(
#     weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
# )


def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    for layer in base_model.layers[:]:
        layer.trainable = False

    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation="relu")(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(CLASS_COUNT, activation="softmax")(custom_model)

    return Model(inputs=input, outputs=predictions)


model = model_maker()
model.summary()

# генератор для обучающей выборки
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # нормализация данных
    rotation_range=40,  # поворот 40 градусов
    width_shift_range=0.2,  # смещенние изображения по горизонтали
    height_shift_range=0.2,  # смещенние изображения по вертикали
    shear_range=0.2,  # случайный сдвиг
    zoom_range=0.2,  # случайное масштабирование
    horizontal_flip=True,  # отражение по горизонтали
    fill_mode="nearest",  # стратегия заполнения пустых пикселей при трансформации
)
# генератор для проверочной выборки
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# генерация картинок из папки для обучающей выборки
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=20,
    class_mode="categorical",
)

# генерация картинок из папки для проверочной выборки
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=20,
    class_mode="categorical",
)


# %%

# компиляция модели
# model.compile(
#     loss="binary_crossentropy",
#     optimizer=optimizers.RMSprop(learning_rate=2e-5),
#     metrics=["acc"],
# )
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"],
)

# обучаем модель
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
)


# %%
# # Функция сглаживания
def smooth_curve(
    points,  # входные точки до сглаживания
    factor=0.8,  # фактор сглаживания
):
    smoothed_points = []  # список из результирующих сглаженных точек
    for point in points:
        if smoothed_points:
            # В условие попадаем если уже в списке есть точки
            previous = smoothed_points[-1]
            # factor = 0.5 - это среднее значение между двумя точками
            # factor можно считать весом при усреднении
            # factor > 0.5 - берем значение ближе к предыдущей точке
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)  # Для первой точки
    return smoothed_points


acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.plot(
    epochs, smooth_curve(acc), "r", label="Сглаженная точность на обучающей выборке"
)
plt.plot(
    epochs,
    smooth_curve(val_acc),
    "b",
    label="Сглаженная точность на проверочной выборке",
)
plt.title("График сглаженных точностей")
plt.legend()
plt.figure()

plt.plot(
    epochs, smooth_curve(loss), "r", label="Сглаженные потери на обучающей выборке"
)
plt.plot(
    epochs,
    smooth_curve(val_loss),
    "b",
    label="Сглаженные потери на проверочной выборке",
)
plt.title("График сглаженных потерь")
plt.legend()
plt.show()

# %%
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="categorical",
    # class_mode="binary"
)

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print("Точность на контрольной выборке:", test_acc)
