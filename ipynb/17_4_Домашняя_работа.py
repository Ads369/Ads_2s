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
# !pip uninstall scikit-learn -y
# !pip install scikit-learn==1.2.2
# !pip install scikeras
# !pip install tensorflow

# %% Import
import matplotlib.pyplot as plt
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import imdb


# %% Main block
# Load and preprocess data
def load_and_preprocess_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=DICT_SPACE
    )

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
def create_model():
    model = models.Sequential(
        [
            layers.Input(shape=(DICT_SPACE,)),
            layers.Dense(
                LAYER_SIZE, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(
                LAYER_SIZE, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(LAYER_SIZE, activation="relu"),
            layers.Dropout(DROPOUT_RATE),
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
def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    oos_y = []
    oos_pred = []
    fold_scores = []
    for fold, (train_index, val_index) in enumerate(kf.split(x_train, y_train), 1):
        print(f"Training fold {fold}")

        _x_train, _x_val = x_train[train_index], x_train[val_index]
        _y_train, _y_val = y_train[train_index], y_train[val_index]

        model = create_model()
        history = train_model(model, _x_train, _y_train, _x_val, _y_val)

        if fold == 1:
            print(model.summary())
            plot_history(history)

        val_pred, val_score = evaluate_model(model, _x_val, _y_val)
        fold_scores.append(val_score)
        print(f"Fold {fold} validation accuracy: {val_score:.4f}")

        # Заполняем списки (реальными и предсказанными) данными, по которым не училась модель
        oos_y.append(_y_val)
        oos_pred.append(val_pred)
    print(f"\n---\nMean accuracy: {np.mean(fold_scores):.4f}")

    _, test_score = evaluate_model(model, x_test, y_test)
    print(f"\nTest accuracy: {test_score:.4f}")

    # Вычисляем ошибку предсказания на всей накопленной в фолдах контрольной выборке.
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    score = accuracy_score(oos_y, oos_pred)
    print(f"Итоговый score (accuracy): {score}")

    # Check goal achievements
    val_accuracy = np.mean(fold_scores)
    if val_accuracy > 0.90:
        print("Task 1")
    if test_score > 0.88:
        print("Task 2")
    if val_accuracy > 0.95 or test_score > 0.93:
        print("Bonus task")

# %% Constants
# Constants
### THIS GOOD CASE BUT NOT ENOUGH
# EPOCHS = 10
# BATCH_SIZE = 128
# DICT_SPACE = 10000
# NUM_FOLDS = 5
# LAYER_SIZE = 16
# DROPOUT_RATE = 0.5

# EXPEREMENTAL ZONE
EPOCHS = 20
BATCH_SIZE = 512
DICT_SPACE = 30000
NUM_FOLDS = 7
LAYER_SIZE = 4
DROPOUT_RATE = 0.6

# %% Run
main()
