# %% Cell 1: Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

word_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}

# %% Cell 2: Load and prepare data
# Load and prepare data
csv_path = Path("../assets/A_Z_Handwritten_Data.csv")
dataset = np.loadtxt(csv_path, delimiter=",")

X = dataset[:, 1:785]
y = dataset[:, 0].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Cell 3: Normalize and reshape data
# Normalize and reshape data
X_train = X_train.reshape((-1, 28, 28, 1)).astype("float32") / 255
X_test = X_test.reshape((-1, 28, 28, 1)).astype("float32") / 255

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 26)
y_test = keras.utils.to_categorical(y_test, 26)

# %% Cell 4: Define model
# Define model
model = keras.Sequential(
    [
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(26, activation="softmax"),
    ]
)

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# %% Cell 5: Train model
# Train model
history = model.fit(
    X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# %% Cell: Predict
# Выбор нужной картинки из тестовой выборки
n = 23
x = X_test[n]
y = np.argmax(y_test[n])

# Проверка формы данных
x = x.reshape((1, 28 * 28, 1))
print(x.shape)

# %% cell: predict
# Предсказываем выбранную картинку
prediction = model.predict(x)

# Вывод результата - вектор из 10 чисел
print(f"Вектор результата на 10 выходных нейронах: {prediction}")

# Получение и вывод индекса самого большого элемента (это значение цифры, которую распознала сеть)
pred = np.argmax(prediction)
print(f"Распознана цифра: {pred}")
print(f"Правильное значение: {np.argmax(y_test[n])}")

# %% Cell 6: Visualize
# Визуализация результата
x = x.reshape((28, 28))
plt.axis("off")
im = plt.subplot(1, 1, 1)
plt.title(word_dict.get(y))
im.imshow(x, cmap="gray")
