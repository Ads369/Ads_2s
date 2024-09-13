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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/17_4_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="SSPwDlw-MmJo"
# **Навигация по уроку**
#
# 1. [Проблема переобучения нейронных сетей](https://colab.research.google.com/drive/10jxb_tGNTenMkxuC7ksTjjUtq966mZMM#scrollTo=SOSDEbSwa-Hj)
# 2. [Методы оптимизации и регуляризации НС](https://colab.research.google.com/drive/1VWkA2xBTwWreo3DTdiMkio-RfQIvlXBl)
# 3. [Универсальный алгоритм машинного обучения](https://colab.research.google.com/drive/1hK-5MC4eApd1-tRMppY9X9RLUYeMqG4q)
# 4. Домашняя работа

# %% [markdown] id="xNAPqodeKiw4"
# Используя знания данного урока, и набор данных IMDB вам необходимо:
# 1. Спроектировать модель классификации отзывов к фильмам с точностью на валидационной выборке более 90%.
# 2. Показать, что модель способна классифицировать отзывы с вероятностью более 88% на контрольной выборке.
#
# За успешное выполнение задания вы получите 3 балла.
# Если сможете преодолеть точность 95% на валидационной выборке
# и/или 93% на контрольной, то получите 4 балла.
#
# Также вы можете получить дополнительно 1 балл, если выполните все предложенные задания в задаче о Титанике (17.1), проанализируете "увеличенную модель" (17.2).

# %% pip
""" "
!pip install scikeras
"""

# %% Import
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# %% Main
def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute.")
        return result

    return wrapper


def preprocess_text(text):
    # import nltk
    # nltk.download("punkt_tab")
    # nltk.download("stopwords")
    # nltk.download("wordnet")

    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def resplit_dataset(train_data, test_data, train_labels, test_labels):
    all_data = np.concatenate((train_data, test_data))
    all_labels = np.concatenate((train_labels, test_labels))

    # Resplit the data 80/20
    train_data, test_data, train_labels, test_labels = train_test_split(
        all_data, all_labels, test_size=0.4, random_state=42
    )

    return train_data, train_labels, test_data, test_labels


def draw_baance_plot(data_tuple):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=data_tuple)
    plt.title("Distribution of Labels in Training Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()


def calculate_balance(data_tuple, do_print: bool = True):
    unique, counts = np.unique(data_tuple, return_counts=True)
    balance = dict(zip(unique, counts))
    pro_balance = balance[1] / balance[0]

    if do_print:
        print("Balance of labels in training data:")
        print(f"Totall: {len(data_tuple)}")
        print(f"0 (Negative): {balance[0]}")
        print(f"1 (Positive): {balance[1]}")
        print(f"Ratio (Positive/Negative): {pro_balance:.2f}")
        print("---")

    if pro_balance < 0.9:
        raise ValueError("The ratio of positive to negative labels is too low.")


def parsing_review():
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    categories = tuple(word_index.keys())

    def sequence_to_text(sequence):
        return " ".join([reverse_word_index.get(i - 3, "?") for i in sequence])

    def text_to_sequence(text):
        return [word_index.get(word, 0) + 3 for word in text.split()]

    print("start prepocessor")
    preprocessed_data = []
    for review in all_data:
        text = sequence_to_text(review)
        preprocessed_data.append(text_to_sequence(text))
        # preprocessed_data.append(text)

    #     from sklearn.preprocessing import LabelEncoder
    #     label_encoder = LabelEncoder()
    #     text = label_encoder.fit_transform(categories)
    #     # preprocessed_text = preprocess_text(text)
    # preprocessed_data.append(text_to_sequence(preprocessed_text))

    print("start tokening")
    # Tokenize and pad sequences
    # tokenizer = Tokenizer(num_words=DICT_SPACE)
    # tokenizer.fit_on_texts(preprocessed_data)
    # sequences = tokenizer.texts_to_sequences(preprocessed_data)
    # x_data = pad_sequences(sequences)


def vectorize_sequences(sequences, dimension=None):
    if dimension is None:
        dimension = DICT_SPACE

    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0  # Записываем единицы в элемент с данным индексом
    return results


def resplit_data(
    train_data, train_labels, test_data, test_labels, test_size=0.2, random_state=42
):
    all_data = np.concatenate((train_data, test_data))
    all_labels = np.concatenate((train_labels, test_labels))

    train_data, test_data, train_labels, test_labels = train_test_split(
        all_data, all_labels, test_size=test_size, random_state=random_state
    )

    return train_data, train_labels, test_data, test_labels


def load_and_preprocess_data(test_size=None):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=DICT_SPACE,
        skip_top=SKIP_TOP,
    )

    if test_size is not None:
        train_data, train_labels, test_data, test_labels = resplit_data(
            train_data, train_labels, test_data, test_labels, test_size=test_size
        )

    # Vectorize for tanserflows
    train_data = vectorize_sequences(train_data)
    test_data = vectorize_sequences(test_data)
    train_labels = np.asarray(train_labels).astype("float32")
    test_labels = np.asarray(test_labels).astype("float32")
    calculate_balance(train_labels)

    return train_data, train_labels, test_data, test_labels


# Create model
def create_model(layer_size=None, dropout_rate=None):
    if layer_size is None:
        layer_size = LAYER_SIZE

    if dropout_rate is None:
        dropout_rate = DROPOUT_RATE

    model = models.Sequential(
        [
            layers.Input(shape=(DICT_SPACE,)),
            layers.Dense(
                layer_size, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(
                layer_size, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Train model
def train_model(model, x, y, x_val, y_val):
    history = model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=VERBOSE_MODEL,
    )
    return history


# Evaluate model
def evaluate_model(model, x_test, y_test):
    raw_predictions = model.predict(x_test)
    predictions = (raw_predictions > 0.5).astype(int)
    score = accuracy_score(y_test, predictions)
    return predictions, score


# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()


@time_execution
def main_2():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    print("Create model")
    model = create_model()

    print("Train model")
    # history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
    history = train_model(model, x_train, y_train, x_test, y_test)


# Main execution
@time_execution
def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data(test_size=0.2)

    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    oos_y = []
    oos_pred = []
    fold_scores = []
    the_best_score = 0.0
    the_best_model = None
    the_best_history = None
    test_score = None

    old_model = create_model()
    old_history = None

    for fold, (train_index, val_index) in enumerate(kf.split(x_train, y_train), 1):
        print(f"Training fold {fold}")

        _x_train, _x_val = x_train[train_index], x_train[val_index]
        _y_train, _y_val = y_train[train_index], y_train[val_index]

        model = create_model()

        history = train_model(model, _x_train, _y_train, _x_val, _y_val)

        # Обучение общей модели
        old_history = train_model(old_model, _x_train, _y_train, _x_val, _y_val)

        if fold == 1:
            print(model.summary())
            plot_history(history)

        # Predict and evaluate
        val_pred, val_score = evaluate_model(model, _x_val, _y_val)

        fold_scores.append(val_score)
        print(f"Fold {fold} validation accuracy: {val_score:.4f}")

        # Заполняем списки (реальными и предсказанными) данными, по которым не училась модель
        oos_y.append(_y_val)
        oos_pred.append(val_pred)

        if the_best_score < val_score:
            the_best_model = model
            the_best_score = val_score
            the_best_history = history

    # Case 3 - model
    final_model = create_model()
    new_history = train_model(final_model, x_train, y_train, x_test, y_test)
    _, new_test_score = evaluate_model(final_model, x_test, y_test)

    print("\n---")
    print(f"\nMean accuracy: {np.mean(fold_scores):.4f}")
    print(f"The best accuracy: {the_best_score:.4f}")
    print("\n---")

    # CASE 1
    # Выбрать модель с наивысшим score в качестве окончательной модели.
    if the_best_model is not None and the_best_history is not None:
        _, best_test_score = evaluate_model(the_best_model, x_test, y_test)
        print(f"\nThe Best Model test accuracy: {best_test_score:.4f}")
        print(
            f"Final validation accuracy: {the_best_history.history['val_accuracy'][-1]:.4f}"
        )
    else:
        print("\nThere is not the Best Model")
    print("---")

    # Case 2 - ?
    # Предварительно установить новые данные для пяти моделей
    # (по одному для каждого фолда) и усреднить результат.
    # Вычисляем ошибку предсказания на всей накопленной в фолдах контрольной выборке.
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    score = accuracy_score(oos_y, oos_pred)
    print(f"\nИтоговый score (accuracy): {score}")
    print("---")

    # Case 3
    # Обучить новую модель на всём наборе данных, используя те же настройки (гиперпараметры),
    # что и при перекрестной проверке: то же число эпох и та же структура слоёв.
    print(f"\nThe New Model accuracy: {new_test_score:.4f}")
    print(f"Final validation accuracy: {new_history.history['val_accuracy'][-1]:.4f}")
    print("---")

    # Case 4
    # Проверка одной, модели которая училась на K-fold'aх
    _, old_test_score = evaluate_model(the_best_model, x_test, y_test)
    print(f"\nThe Old Model test accuracy: {old_test_score:.4f}")
    if old_history is not None:
        print(
            f"Final validation accuracy: {old_history.history['val_accuracy'][-1]:.4f}"
        )
    print("---")

    # Check goal achievements
    val_accuracy = np.mean(fold_scores)
    if val_accuracy > 0.90:
        print("Task 1")

    if test_score:
        if test_score > 0.88:
            print("Task 2")
        if val_accuracy > 0.95 or test_score > 0.93:
            print("Bonus task")


# %% Cell grid search
from scikeras.wrappers import KerasClassifier


@time_execution
def main_grid_search():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Создание KerasClassifier
    model = KerasClassifier(
        model=create_model, layer_size=LAYER_SIZE, dropout_rate=DROPOUT_RATE, verbose=0
    )

    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Определение диапазонов гиперпараметров для поиска
    param_grid = {
        # "layer_size": [4, 16, 32, 64, 128],
        # 'optimizer': ['adam', 'rmsprop'],
        "layer_size": [4, 16, 32],
        "dropout_rate": [0.3, 0.5, 0.7],
        "batch_size": [32, 64, 128, 256],
        "epochs": [5, 10, 15],
    }

    if np.prod([len(v) for v in param_grid.values()]) > 10:
        print("Using RandomizedSearchCV")
        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=SCORING,
            cv=kf,
            n_iter=10,
            random_state=42,
            verbose=1,
        )
    else:
        print("Using GridSearchCV")
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=SCORING,
            cv=kf,
            verbose=3,
        )

    # For debug
    # print("Available parameters:")
    # print(model.get_params().keys())

    # Запуск GridSearchCV
    grid_result = grid.fit(x_train, y_train)

    # Отображение лучших результатов
    print(f"Лучшие параметры: {grid_result.best_params_}")
    print(f"Лучшая точность: {grid_result.best_score_:.4f}")

    # Оценка модели с лучшими параметрами на тестовой выборке
    best_model = grid_result.best_estimator_
    test_score = best_model.score(x_test, y_test)
    print(f"Точность на тестовой выборке: {test_score:.4f}")


# %% Constants
# Constants
SCORING = "accuracy"
PRE_TRAINED_MODEL_NAME = "bert-base-cased"
MAX_LEN = 400
DICT_SPACE = 10000
SKIP_TOP = 2
VERBOSE_MODEL = 0

# It's  good work don't touch
# EPOCHS = 10
# BATCH_SIZE = 128
# DICT_SPACE = 10000
# NUM_FOLDS = 10
# LAYER_SIZE = 16
# DROPOUT_RATE = 0.5

# Experemental zone
EPOCHS = 10
BATCH_SIZE = 128
NUM_FOLDS = 10
LAYER_SIZE = 32
DROPOUT_RATE = 0.5


# %% Run
main()
# main_2()
# main_grid_search()
# load_and_preprocess_data()

# %% markdown
# Проведя исследования
#
# Гиперпараметры котороые я получил ручным перебором
# EPOCHS = 10
# BATCH_SIZE = 128
# NUM_FOLDS = 5
# LAYER_SIZE = 16
# DROPOUT_RATE = 0.5
#
# Гипер параметры которые я получили с помощью GridSearchCV
# EPOCHS = 10
# BATCH_SIZE = 128
# NUM_FOLDS = 5
# LAYER_SIZE = 32
# DROPOUT_RATE = 0.5
#
# Mean accuracy: 0.8780
# The best accuracy: 0.8834
# ---
# The Best Model test accuracy: 0.8706
# Final validation accuracy: 0.8834
# ---
# Итоговый score (accuracy): 0.878
# ---
# The New Model accuracy: 0.8706
# Final validation accuracy: 0.8705
# ---
# The Old Model test accuracy: 0.8706
# Final validation accuracy: 0.9058
#
# Однако полученного результата не достаточно что-бы выполнить ДЗ
#
# При ресайзе 80/20 значения я получаю следующие значения
# Mean accuracy: 0.8821
# The best accuracy: 0.8884
# ---
# The Best Model test accuracy: 0.8838
# Final validation accuracy: 0.8884
# ---
# Итоговый score (accuracy): 0.8821
# ---
# The New Model accuracy: 0.8838
# Final validation accuracy: 0.8892
# ---
# The Old Model test accuracy: 0.8838
# Final validation accuracy: 0.8986
# ---
#
# Я хотел попробовать реализовать обработку строк:
# - У даление часто повторяющихся слов
# - Удаление не нужных символов
# - Приведене слов в простой форме
# Но с данной задачей ни мой ПК ни Colab справиться не может
# Мой финальный результат
# Mean accuracy: 0.8857
# The best accuracy: 0.8950
# ---
# The Best Model test accuracy: 0.8882
# Final validation accuracy: 0.8950
# ---
# Итоговый score (accuracy): 0.885675
# ---
# The New Model accuracy: 0.8882
# Final validation accuracy: 0.8899
# ---
# The Old Model test accuracy: 0.8882
# Final validation accuracy: 0.9115
# ---
