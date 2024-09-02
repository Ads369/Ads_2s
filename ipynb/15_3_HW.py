# %% Cell 0: Imports and Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Constants
RANDOM_SEED = 42
NUM_CLASSES = 26
WORD_DICT = {i: chr(65 + i) for i in range(NUM_CLASSES)}
IMG_SIZE = 28
IMG_VECTOR = IMG_SIZE**2
history = None

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# %% Cell 1: Load and Prepare Data
def load_and_prepare_data(file_path):
    dataset = np.loadtxt(file_path, delimiter=",")
    x = dataset[:, 1 : IMG_VECTOR + 1].astype(np.float32)
    y = dataset[:, 0].astype(np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return x_train, x_test, y_train, y_test


csv_path = Path("../assets/A_Z_Handwritten_Data.csv")
x_train, x_test, y_train, y_test = load_and_prepare_data(csv_path)

print(f"Train shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")


# %% Cell : check balance
# for i in range(26):
#     print(f"Class {i} train: {np.mean(y_train[:, i])}, test: {np.mean(y_test[:, i])}")
# draw diagramma of class balance
plt.plot(np.mean(y_train, axis=0), label="Train")
plt.plot(np.mean(y_test, axis=0), label="Test")
plt.legend()
plt.show()


# Cell: Define and Compile Model
def create_dense(layer_sizes):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(IMG_VECTOR,)))

    for s in layer_sizes:
        model.add(layers.Dense(s, activation="sigmoid"))

    model.add(layers.Dense(units=NUM_CLASSES, activation="softmax"))
    return model


def evaluate(model, batch_size=128, epochs=5):
    model.summary()
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=False,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["training", "validation"], loc="best")
    plt.show()

    print()
    print(f"Test loss: {loss:.3}")
    print(f"Test accuracy: {accuracy:.3}")
    return history


# %% Cell 3: Train Model
# Проверка по количеству слоев
for _layers in range(1, 5):
    model = create_dense([32] * _layers)
    history = evaluate(model)

# %% MarkDown
# Тестирование по количкству слоев показало, cебя плохо.
# Больше 2 слоев - ухудшение качества результата.

# %% Cell
# Проверка по ширине слоя
for nodes in [32, 64, 128, 256, 512, 1024, 2048]:
    model = create_dense([nodes])
    history = evaluate(model)

# %% MarkDown
# Тестирование по ширине слоя показало себя странно.
# Модель явно улучшает качество результата.
# Но график показывать высокую вероятность переобучаесаемости.


# %% Cell
# Проверка по количкситву эпох
for evals in range(1, 5):
    model = create_dense([512, 256])
    history = evaluate(model, epochs=evals)

# %% MarkDown
# Тестирование по эпохам показало себя странно.
# Модель явно улучшает качество результата.
# Но график показывать высокую вероятность переобучаесаемости.


# %% Cell 5: Visualize Results
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if history is not None:
    plot_history(history)
else:
    print("No history")


# %% Cell 6: Predictions and Visualization
def visualize_prediction(model, x_test, y_test, index):
    x = x_test[index].reshape(1, -1)
    y_true = np.argmax(y_test[index])

    prediction = model.predict(x)
    y_pred = np.argmax(prediction)

    plt.figure(figsize=(6, 6))
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {WORD_DICT[y_pred]}, True: {WORD_DICT[y_true]}")
    plt.axis("off")
    plt.show()


# Visualize a random prediction
random_index = np.random.randint(0, len(x_test))
visualize_prediction(model, x_test, y_test, random_index)
