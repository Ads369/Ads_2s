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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/ipynb/Lesson%2026/26_4_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="KF4Hi0OeNSYn"
# **Навигация по уроку**
#
# 1. [Веб-архитектура сервиса](https://colab.research.google.com/drive/10wtDodlf4SaVcYk6VoXDWk650IDcNPaa)
# 2. [Практическое использование REST API в Python](https://colab.research.google.com/drive/1bhlFqhZp0TtOuzqKJvI9C-K0FwRWMi2H)
# 3. [Введение в FastAPI](https://colab.research.google.com/drive/1_AzAVys4xub3yyw763NDwfeJ3WecGgkb)
# 4. Домашняя работа

# %% [markdown] id="Ogb5_BJzQjOZ"
# В домашней работе вам необходимо с помощью **FastAPI** реализовать **REST API**:
#
# 1. На 3 балла. Ваш REST API - это список покупок и содержит поля: название товара, группа товара (например, электроника или продовольствие), цена, единица измерения, количество. Также необходимо реализовать метод, который возвращает список - расходы по каждой группе товаров и сумму всех покупок.
#
#   Также необходимо с помощью библиотеки `requests` продемонстрировать запросами к REST API, как работает ваш веб-сервис. Это задание можно сравнить с "покрытием тестами" вашего API. Нечто похожее делают тестировщики в ИТ-компаниях. Вам необходимо покрыть запросами все методы, которые вы реализуете на веб-сервере.
#
# 2. На 4 балла. Вам необходимо сделать красивую документацию для вашего REST API с подробным описанием. Для этого вам придется обратиться к документации:
#   * https://fastapi.tiangolo.com/ru/tutorial/metadata/
#   * https://fastapi.tiangolo.com/ru/tutorial/path-operation-configuration/#response-description
# 3. На 5 баллов. Творческое задание. REST API можно использовать для взаимодействия с вашей моделью нейронной сети. Вы уже знаете, что можно обучить модель, а лучший результат выгрузить для дальнейшего использования. Для получения 5 баллов необходимо обучить свою модель, загрузить ее в Colab. Задача может быть любой: регрессии, классификации, входными данными могут быть картинки или текстовые данные. С помощью REST API обеспечьте взаимодействие с моделью. Это полностью творческое задание!
#
#


# %%
# !pip install keras-tuner -q
from enum import Enum
from typing import Dict, List


import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import uvicorn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt



# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Conv2D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('kernel_size', values=[3, 5]),
        activation='relu',
        input_shape=(28, 28, 1)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('units', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='mnist_automl',
    project_name='mnist'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of filters is {best_hps.get('filters')},
the best kernel size is {best_hps.get('kernel_size')},
the best number of units in the Dense layer is {best_hps.get('units')},
the optimal dropout rate is {best_hps.get('dropout')},
and the best learning rate is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

model.save('mnist_model')
print("Model saved to file.")



# %%
# FastAPI
app = FastAPI(
    title="Shopping List API",
    description="A REST API to manage and analyze a shopping list.",
    version="1.0.0",
)


MODEL_PATH = None


class ProductGroup(str, Enum):
    ELECTRONICS = "Electronics"
    GROCERIES = "Groceries"
    CLOTHING = "Clothing"
    BOOKS = "Books"
    HOUSEHOLD = "Household"
    COSMETICS = "Cosmetics"
    TOYS = "Toys"
    SPORTS = "Sports"


class Item(BaseModel):
    name: str
    group: ProductGroup
    price: float
    unit: str
    quantity: int


shopping_list: List[Item] = []


@app.get(
    "/product-groups", response_model=List[str], summary="Get available product groups"
)
async def get_product_groups():
    """Return a list of available product groups."""
    return [group.value for group in ProductGroup]


@app.post("/items", response_model=Item, summary="Add a new item")
async def add_item(item: Item):
    """
    Add a new item to the shopping list.

    The shopping list items contain:
    - **name**: Name of the product
    - **group**: Product category/group (_use GET /product-groups for it_)
    - **price**: Price per unit
    - **unit**: Unit of measurement
    - **quantity**: Number of items
    """
    shopping_list.append(item)
    return item


@app.get("/items", response_model=List[Item], summary="Get all items")
async def get_items():
    """Retrieve all items in the shopping list."""
    return shopping_list


@app.get(
    "/costs", response_model=Dict[str, float], summary="Get costs by product group"
)
async def get_costs():
    """Calculate and return total costs for each product group and overall total."""
    costs: Dict[str, float] = {}
    total_sum = 0.0
    for item in shopping_list:
        group_total = item.price * item.quantity
        costs[item.group] = costs.get(item.group, 0) + group_total
        total_sum += group_total
    costs["Total"] = total_sum
    return costs


@app.post("/predict", summary="Make MNIST prediction")
async def predict_digit(image: bytes):
    """
    Make a prediction on a digit image using MNIST model.
    Returns predicted digit class and confidence score.
    """

    if MODEL_PATH is None:
        raise HTTPException(
            status_code=500,
            detail="MNIST model not loaded. Please ensure model is available.",
        )

    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Preprocess image
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))

        # Make prediction
        prediction = MODEL_PATH.predict(img)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])

        return {"predicted_digit": int(predicted_class), "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Custom OpenAPI documentation
@app.get("/docs", summary="API Documentation", include_in_schema=False)
def custom_docs():
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="Shopping List API",
            version="1.0.0",
            description="A detailed API documentation for managing a shopping list and analyzing costs.",
            routes=app.routes,
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    return custom_openapi()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# %%
# Example test coverage using the requests library
def test_api():
    base_url = "http://127.0.0.1:8000"

    # Add items
    item_1 = {
        "name": "Laptop",
        "group": "Electronics",
        "price": 1000,
        "unit": "piece",
        "quantity": 1,
    }
    item_2 = {
        "name": "Apples",
        "group": "Groceries",
        "price": 2,
        "unit": "kg",
        "quantity": 3,
    }

    r1 = requests.post(f"{base_url}/items", json=item_1)
    r2 = requests.post(f"{base_url}/items", json=item_2)
    print("Add items:", r1.json(), r2.json())

    # Get all items
    r3 = requests.get(f"{base_url}/items")
    print("All items:", r3.json())

    # Get costs
    r4 = requests.get(f"{base_url}/costs")
    print("Costs:", r4.json())
