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
# За успешное выполнение задания вы получите 3 балла. Если сможете преодолеть точность 95% на валидационной выборке и/или 93% на контрольной, то получите 4 балла.
#
# Также вы можете получить дополнительно 1 балл, если выполните все предложенные задания в задаче о Титанике (17.1), проанализируете "увеличенную модель" (17.2).

# %% pip
#

# %% Import
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import imdb


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
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # # Tokenize
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
        all_data, all_labels, test_size=0.2, random_state=42
    )

    return train_data, train_labels, test_data, test_labels


def draw_baance_plot(data_tuple):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=data_tuple)
    plt.title("Distribution of Labels in Training Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()


def calculate_balance(data_tuple):
    unique, counts = np.unique(data_tuple, return_counts=True)
    balance = dict(zip(unique, counts))
    print("Balance of labels in training data:")
    print(f"Totall: {len(data_tuple)}")
    print(f"0 (Negative): {balance[0]}")
    print(f"1 (Positive): {balance[1]}")
    print(f"Ratio (Positive/Negative): {balance[1]/balance[0]:.2f}")
    print("---")


# Load and preprocess data
def load_and_preprocess_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=DICT_SPACE,
        skip_top=20,
    )

    train_data, train_labels, test_data, test_labels = resplit_dataset(
        train_data, test_data, train_labels, test_labels
    )

    calculate_balance(train_labels)

    def vectorize_sequences(sequences, dimension=DICT_SPACE):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    return x_train, y_train, x_test, y_test


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
            # layers.Dense(
            #     layer_size, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            # ),
            # layers.Dropout(dropout_rate),
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
        verbose=0,
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


# Main execution
@time_execution
def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

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

    print("\n---")
    print(f"\nMean accuracy: {np.mean(fold_scores):.4f}")
    print(f"The best accuracy: {the_best_score:.4f}")
    print("\n---")

    # CASE 1
    # Выбрать модель с наивысшим score в качестве окончательной модели.
    if the_best_model is not None and the_best_history is not None:
        _, test_score = evaluate_model(the_best_model, x_test, y_test)
        print(f"\nThe Best Model test accuracy: {test_score:.4f}")
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
    final_model = create_model()
    _history = train_model(final_model, x_train, y_train, x_test, y_test)
    _, test_score = evaluate_model(final_model, x_test, y_test)
    print(f"\nThe New Model accuracy: {test_score:.4f}")
    print(f"Final validation accuracy: {_history.history['val_accuracy'][-1]:.4f}")
    print("---")

    # Case 4
    # Проверка одной, модели которая училась на K-fold'aх
    _, test_score = evaluate_model(the_best_model, x_test, y_test)
    print(f"\nThe Old Model test accuracy: {test_score:.4f}")
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

# It's  good work don't touch
# EPOCHS = 10
# BATCH_SIZE = 128
# DICT_SPACE = 10000
# NUM_FOLDS = 5
# LAYER_SIZE = 16
# DROPOUT_RATE = 0.5

# Experemental zone
EPOCHS = 10
BATCH_SIZE = 128
DICT_SPACE = 20000
NUM_FOLDS = 5
LAYER_SIZE = 32
DROPOUT_RATE = 0.5

# %% Run
main()
# main_grid_search()
# load_and_preprocess_data()
