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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/22_3_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="Wscc0Fl5qhuR"
#  **Навигация по уроку**
#
# 1. [Решение задач регрессии с помощью НС](https://colab.research.google.com/drive/1GbXbqPbC4A2NVJEj-5pOOySN3gyalBQN)
# 2. [Анализ резюме кандидатов](https://colab.research.google.com/drive/1L4pI4giYvWY3T4gfqTF5c_XFkN9k4Nfz)
# 3. Домашняя работа

# %% [markdown] id="_up7JAoo8CF0"
# **В домашней работе необходимо выполнить следующее задание:**
#
# 1. Используя предложенный [датасет](https://storage.yandexcloud.net/academy.ai/japan_cars_dataset.csv) японских машин, обучите модель предсказывать цены на японские автомобили.
# 2. Создайте обучающую, тестовую и проверочную выборки.
# 3. Оцените качество работы созданной сети, определите средний процент ошибки на проверочной выборке.
# 4. В качестве ошибки рекомендуется использовать среднеквадратическую ошибку (mse).
# 5. Выполнив задание, получите 3 балла.
# 6. Хотите 4 балла? Добейтесь ошибки менее 10%.
# 7. Хотите 5 баллов? Добейтесь ошибки менее 5%.
#

# %% [markdown] id="7b9zW5YN6znk"
# **Примечание**. Подробную информацию о датасете можно узнать на портале соревновани [kaggle.com](https://www.kaggle.com/datasets/doaaalsenani/used-cars-dataets/data).
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="BDCOnE5A7XiG" outputId="c6c455d1-0d56-4c8d-cdfd-a281ae0c0836"
import requests
from sklearn.metrics import mean_absolute_error

url = "https://storage.yandexcloud.net/academy.ai/japan_cars_dataset.csv"
response = requests.get(url)
with open("japan_cars_dataset.csv", "wb") as f:
    f.write(response.content)

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="NS_vFnv17mjE" outputId="99be074f-2971-4b03-855f-cbfc855c6347"
import pandas as pd

cars = pd.read_csv("japan_cars_dataset.csv", sep=",")
# cars.head(10)
cars.info()

# %% id="b-a8LLHThFg8"
# ваше решение


# %% cell
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# %% cell
# Load and preprocess data
def prepare_data(file_path):
    """
    Prepare car data for neural network training.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        tuple: Contains:
            - categorical_encoded: One-hot encoded categorical features
            - numerical_scaled: Scaled numerical features
            - prices_scaled: Scaled target prices
            - numerical_scaler: Scaler for numerical features
            - price_scaler: Scaler for prices
    """
    # Read and clean data
    cars = pd.read_csv(file_path)
    cars = cars.dropna()

    # Split features into categorical and numerical
    categorical_columns = [
        "mark",
        "model",
        "transmission",
        "drive",
        "hand_drive",
        "fuel",
    ]
    label_encoders = {}
    categorical_encoded = []
    numerical_columns = ["year", "mileage", "engine_capacity"]

    categorical_features = cars[categorical_columns]
    encoder = OneHotEncoder()
    categorical_encoded = encoder.fit_transform(categorical_features)

    # for column in categorical_columns:
    #     label_encoders[column] = LabelEncoder()
    #     encoded_data = label_encoders[column].fit_transform(cars[column])
    #     categorical_encoded.append(encoded_data)

    numerical_features = cars[numerical_columns]
    numerical_scaler = StandardScaler()
    numerical_scaled = numerical_scaler.fit_transform(numerical_features)

    price_scaler = StandardScaler()
    prices_scaled = price_scaler.fit_transform(cars[["price"]])

    return (
        categorical_encoded,
        numerical_scaled,
        prices_scaled,
        price_scaler,
    )


# %% cell
# Create model
def create_model(categorical_shape, numerical_shape):
    # Categorical input branch
    categorical_input = Input(shape=(categorical_shape,))
    x1 = Dense(64, activation="relu")(categorical_input)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(32, activation="relu")(x1)

    # Numerical input branch
    numerical_input = Input(shape=(numerical_shape,))
    x2 = Dense(32, activation="relu")(numerical_input)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(16, activation="relu")(x2)

    # Combine branches
    combined = concatenate([x1, x2])

    # Output layers
    x = Dense(64, activation="relu")(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x)

    model = Model(inputs=[categorical_input, numerical_input], outputs=x)
    return model


def split_data(dataset_parts, test_size=0.2):
    """
    Split data into training, validation and test sets.

    Args:
        dataset_parts (tuple): Tuple containing parts of dataset to split
        test_size (float): Size of test set as fraction of total dataset (default 0.3)

    Returns:
        tuple: Split datasets with same order as input
    """
    train_parts = []
    temp_parts = []
    val_parts = []
    test_parts = []

    for part in dataset_parts:
        train, temp = train_test_split(part, test_size=test_size, random_state=42)
        train_parts.append(train)
        temp_parts.append(temp)

    for temp in temp_parts:
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        val_parts.append(val)
        test_parts.append(test)

    result = []
    for train, val, test in zip(train_parts, val_parts, test_parts):
        result.extend([train, val, test])

    return tuple(result)


# %% cell
# # Training and evaluation
def train_and_evaluate(
    categorical_data,
    numerical_data,
    y_train,
    cat_val,
    num_val,
    y_val,
    epochs=10,
    batch_size=32,
):
    model = create_model(categorical_data.shape[1], numerical_data.shape[1])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])

    # Train model
    history = model.fit(
        [cat_train, num_train],
        y_train,
        validation_data=([cat_val, num_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Evaluate model
    test_predictions = model.predict([cat_test, num_test])

    # Calculate error metrics
    mse = np.mean((y_test - test_predictions) ** 2)
    mae = np.mean(np.abs(y_test - test_predictions))
    mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

    print("\nTest Set Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Model MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, price_scaler, mape


# %% cell
# Prepare data
categorical_data, numerical_data, prices, price_scaler = prepare_data(
    "japan_cars_dataset.csv"
)
split_datasets = split_data([categorical_data, numerical_data, prices])

# Unpack the split datasets
cat_train, cat_val, cat_test = split_datasets[0:3]
num_train, num_val, num_test = split_datasets[3:6]
y_train, y_val, y_test = split_datasets[6:9]

print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_val)}")
print(f"Test set size: {len(y_test)}")

# %% cell
# Run training and evaluation
model, price_scaler, error_percentage = train_and_evaluate(
    cat_train, num_train, y_train, cat_val, num_val, y_val
)


# %% cell
# Предсказание на новых данных (контрольный образец)
pred_test = model.predict([cat_test, num_test])
# %% cell
# Предсказание на новых данных (контрольный образец)
pred_test = model.predict([cat_test, num_test])
mean_error = mean_absolute_error(pred_test, y_test)
print("Средняя абсолютная ошибка:", mean_error)

# %% cell
for i in range(10):
    x = y_test[i, 0]
    y = pred_test[i, 0]
    print(
        "Реальное значение: {:6.2f}  Предсказанное значение: {:6.2f}  Разница: {:6.2f}".format(
            x,
            y,
            abs(x - y),
        )
    )


# %% cell
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pred_test, y_test)  # Отрисовка точечного графика
ax.set_xlim(0, 1)  # Ограничение оси по x
ax.set_ylim(0, 1)  # Ограничение оси по x
ax.plot(plt.xlim(), plt.ylim(), "r")  # Отрисовка диагональной линии
plt.xlabel("Правильные значения")
plt.ylabel("Предсказания")
plt.grid()
plt.show()
