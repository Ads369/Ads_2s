# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/15_3_%D0%94%D0%BE%D0%BC%D0%B0%D1%88%D0%BD%D1%8F%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="yvUS7h_FGb0U"
# **Навигация по уроку**
#
# 1. [Простые нейронные сети. Знакомство с библиотекой Keras](https://colab.research.google.com/drive/16xfRAdyg-Re1bP2cyYpbpAALa6noUL3U)
# 2. [Распознание рукописных цифр (Практика)](https://colab.research.google.com/drive/1RaGkCutdIazFN5PtQOod5UH2Cc05diYI)
# 3. Домашняя работа

# + [markdown] id="xrPK-wzwLQnp"
# Используя датасет по рукописным буквам английского языка, обучите модель, оцените ее предсказательные способности. Используйте только полносвязанные слои. Поэкспериментируйте с числом слоев и числом нейронов в слое, добейтесь максимальной точности. Используйте куски кода и рекомендации из практической части урока. Нарисуйте графики точности и потерь для обучающей и тестовой выборки, сделайте по ним выводы.
#
# Оценка за задание:
# * 1 балл - задача решена с помощью куратора, точность на проверочной выборке ниже 85%
# * 2 балла - задача решена с подсказками куратора, точность на проверочной выборке выше 85%
# * 3 балла - задача решена самостоятельно, точность на проверочной выборке выше 85%
# * 4 балла - задача решена самостоятельно, точность на проверочной выборке выше 93%
# * 5 баллов - задача решена самостоятельно, точность на проверочной выборке выше 97%

# + id="Yh7MYaC4GJXD"
 import numpy as np
dataset = np.loadtxt('https://storage.yandexcloud.net/academy.ai/A_Z_Handwritten_Data.csv', delimiter=',')

# + id="qyJZJs24KxPE"
X = dataset[:,1:785]
Y = dataset[:,0]

# + id="K3Qu65okK53b"
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, shuffle=True)

# + colab={"base_uri": "https://localhost:8080/", "height": 496} id="VVauiGX_OSmi" outputId="5c47c472-5c69-45aa-dd2e-0f9c0cb0f061"

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

for i in range(40):
    x = x_train[i]
    x = x.reshape((28, 28))
    plt.axis('off')
    im = plt.subplot(5, 8, i+1)
    plt.title(word_dict.get(y_train[i]))
    im.imshow(x, cmap='gray')


# + id="y2QBe1cYLRZa"
# Ваше решение
