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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% cell
# Папка с папками картинок, рассортированных по категориям
IMAGE_PATH = "./temp/training_set/training_set/"
IMAGE_PATH_TEST = "./temp/test_set/test_set/"
IMG_WIDTH = 150
IMG_HEIGHT = 150

# Папка в которой будем создавать выборки
BASE_DIR = "./dataset/"

# Определение списка имен классов
CLASS_LIST = sorted(os.listdir(IMAGE_PATH))

# Определение количества классов
CLASS_COUNT = len(CLASS_LIST)

# При повторном запуске пересоздаим структуру каталогов
# Если папка существует, то удаляем ее со всеми вложенными каталогами и файлами
if os.path.exists(BASE_DIR):
    shutil.rmtree(BASE_DIR)

# Создаем папку по пути BASE_DIR
os.mkdir(BASE_DIR)

# Сцепляем путь до папки с именем вложенной папки. Аналогично BASE_DIR + '/train'
train_dir = os.path.join(BASE_DIR, "train")

# Создаем подпапку, используя путь
os.mkdir(train_dir)

# Сцепляем путь до папки с именем вложенной папки. Аналогично BASE_DIR + '/validation'
validation_dir = os.path.join(BASE_DIR, "validation")

# Создаем подпапку, используя путь
os.mkdir(validation_dir)

# Сцепляем путь до папки с именем вложенной папки. Аналогично BASE_DIR + '/test'
test_dir = os.path.join(BASE_DIR, "test")

# Создаем подпапку, используя путь
os.mkdir(test_dir)

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


# Функция создания подвыборок (папок с файлами)
def create_dataset(
    img_path: str,  # Путь к файлам с изображениями классов
    new_path: str,  # Путь к папке с выборками
    class_name: str,  # Имя класса (оно же и имя папки)
    start_index: int = 0,  # Стартовый индекс изображения, с которого начинаем подвыборку
    end_index: int = -1,  # Конечный индекс изображения, до которого создаем подвыборку
):
    src_path = os.path.join(
        img_path, class_name
    )  # Полный путь к папке с изображениями класса
    dst_path = os.path.join(
        new_path, class_name
    )  # Полный путь к папке с новым датасетом класса

    # Получение списка имен файлов с изображениями текущего класса
    class_files = os.listdir(src_path)

    # Создаем подпапку, используя путь
    os.mkdir(dst_path)

    # Перебираем элементы, отобранного списка с начального по конечный индекс
    for fname in class_files[start_index:end_index]:
        # Путь к файлу (источник)
        src = os.path.join(src_path, fname)
        # Новый путь расположения файла (назначение)
        dst = os.path.join(dst_path, fname)
        # Копируем файл из источника в новое место (назначение)
        shutil.copyfile(src, dst)


for class_label in range(
    CLASS_COUNT
):  # Перебор по всем классам по порядку номеров (их меток)
    class_name = CLASS_LIST[class_label]  # Выборка имени класса из списка имен

    create_dataset(IMAGE_PATH, train_dir, class_name, 0, 4000)
    create_dataset(IMAGE_PATH, validation_dir, class_name, 0, 4000)
    create_dataset(IMAGE_PATH_TEST, test_dir, class_name, 0, 1000)

# %% cell
datagen = ImageDataGenerator(
    rescale=1.0 / 255
)  # Задаем генератор и нормализуем данные делением на 255
batch_size = 20  # Размер батча (20 изображений)


# Функция извлечения признаков
def extract_features(directory, sample_count):
    # определяем размерность признаков, заполняем нулями
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    # определяем размерность выходных меток, заполняем нулями
    labels = np.zeros(shape=(sample_count))

    # генерируем данные из папки
    generator = datagen.flow_from_directory(
        directory,  # путь к папке
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # изменить картинки до размера 150 х 150
        batch_size=batch_size,  # размер пакета
        # class_mode="binary",  # задача бинарной классификации
        class_mode="categorical",
    )
    i = 0
    for (
        inputs_batch,
        labels_batch,
    ) in generator:  # в цикле пошагово генерируем пакет с картинками и пакет из меток
        features_batch = model.predict(
            inputs_batch, verbose="0"
        )  # делаем предсказание на сгенерируемом пакете
        features[i * batch_size : (i + 1) * batch_size] = (
            features_batch  # складываем пакеты с признаками пачками в массив с признаками
        )

        # labels[i * batch_size : (i + 1) * batch_size] = (
        #     labels_batch  # складываем пакеты с метками в массив с метками
        # )
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch[:, 0]
        i += 1

        if (
            i * batch_size >= sample_count
        ):  # Прерываем генерацию, когда выходим за число желаемых примеров
            break

    return features, labels  # возвращаем кортеж (признаки, метки)


# Извлекаем (признаки, метки)
train_features, train_labels = extract_features(train_dir, 8000)
validation_features, validation_labels = extract_features(validation_dir, 8000)
test_features, test_labels = extract_features(test_dir, 2000)

# %%
train_features = np.reshape(
    train_features, (8000, 4 * 4 * 512)
)  # приводим к форме (образцы, 8192) обучающие признаки
validation_features = np.reshape(
    validation_features, (8000, 4 * 4 * 512)
)  # приводим к форме (образцы, 8192) проверочные признаки
test_features = np.reshape(
    test_features, (2000, 4 * 4 * 512)
)  # приводим к форме (образцы, 8192) тестовые признаки

# %%

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
    # class_mode="binary"
)

# генерация картинок из папки для проверочной выборки
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=20,
    class_mode="categorical",
    # class_mode="binary"
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
