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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/21_3_%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80_%D1%80%D0%B5%D0%BA%D1%83%D1%80%D1%80%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D1%85_%D0%B8_%D0%BE%D0%B4%D0%BD%D0%BE%D0%BC%D0%B5%D1%80%D0%BD%D1%8B%D1%85_%D1%81%D0%B2%D0%B5%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D1%85_%D1%81%D0%B5%D1%82%D0%B5%D0%B9.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="34eU9bRVr49Y"
# **Навигация по уроку**
#
# 1. [Рекуррентные нейронные сети](https://colab.research.google.com/drive/1Mm5yFeJXZT9YcwlQMGx_T5JcEVgV8ZWy)
# 2. [Одномерные сверточные нейронные сети](https://colab.research.google.com/drive/1SCmcJdfsaxpJiQz_SOMH6gixV-43zPIB)
# 3. Сравнение архитектур рекуррентных и одномерных сверточных сетей
# 4. [Домашняя работа](https://colab.research.google.com/drive/1NMDG3ZeGgyHm0ei0DOC63PBb9rW6oN3O)

# %% [markdown] id="vtym6pFqvjDF"
# Теперь мы можем перейти к практической части урока по рекуррентным и сверточным
# сетям. Мы продолжим решать уже знакомую задачу по классификации текстов русских писателей. Также продолжим использовать навыки Наташи для предобучения слоя `Embedding`. Однако, теперь мы проведем исследование различных архитектур, как настоящие нейронщики, выберем наиболее эффективные слои из рекурентных или одномерных сверточных, а также подберем параметры.

# %% [markdown] id="bf-bpIhoH45X"
# ## Обработка художественных текстов рекуррентными и сверточными сетями

# %% [markdown] id="zW-BBkZA5laP"
# ### Импортируем необходимые модули

# %% id="eUm5MmNM5t3e"
from keras.models import Sequential
from keras.utils import get_file, to_categorical
from keras.layers import Embedding, Flatten, Dense, SpatialDropout1D, BatchNormalization, Dropout, SimpleRNN
from keras.layers import GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

import os   # Модуль для работы с файловой системой

import glob # Вспомогательный модуль для работы с файловой системой
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Для работы с матрицей ошибок

import matplotlib.pyplot as plt
# %matplotlib inline


# %% [markdown] id="6ANZygLPVVXU"
# Мы подгружаем новые для вас слои – **SimpleRNN**, **GRU**, **LSTM**,  **SpatialDropout1D**, **Bidirectional**, **Conv1D**, **MaxPooling1D**, **GlobalMaxPooling1D**.
#
# У **MaxPooling1D** и **GlobalMaxPooling1D** логика работы одинаковая, отличается только в размере окна прохождения. Пусть у нас имеется тензор `[1, 4, 8, 4, 5, 2]` и размер окна 2. При прохождениим  MaxPooling1D получим тензор `[4, 8, 5]`, так как берет максимум в окне. GlobalMaxPooling1D выдаст `[8]`, так как его окно всегда равняется максимально возможному размеру тензора.
#
# **SpatialDropout1D** используется после слоя **Embedding** в случае, если после слоя **Embedding** нужно применить слой **Dropout**. Это вызвано тем, что слой **Embedding** разворачивает слова в таблицу с тензорами определенной длины. И **Dropout**-слой будет выкидывать значения из тензоров, что нарушит логику работы. Слой **SpatialDropout1D** выкидывает целый тензор, что и требуется.
#
# Например, у нас имеется тензор $[[1, 1, 1], [2, 2, 2], [3, 3, 3]]$. Если мы к нему применем **Dropout**, то получим что-то вида $[[1, 0, 1], [0, 2, 2], [[3,0, 3]]]$, где значения обнулены в случайном порядке. Если пропустим тензор через **SpatialDropout1D**, то получим $[[1, 0, 1], [2, 0, 2], [3, 0, 3]]$. Мы видим, что был очищен случайный столбец целиком. Таким образом при обрабоке текстов, **SpatialDropout1D** будет убирать вектор слова целиком.
#
# При создании рекуррентного слоя **SimpleRNN**, **GRU**, **LSTM** всегда указывается размерность скрытого состояния units как именованный параметр:
#
# ```python
# units    = 4          # размерность скрытого состояния
# model = Sequential()
# ...
# model.add(SimpleRNN(units=units))
# ```
# или как позиционный параметр:
#
# ```python
# model.add(SimpleRNN(4))
# ```
# В примерах ниже мы будем использовать задавать функции активации для рекуррентных слоев и параметры регуляризации. Так например:
#
# `GRU(4, dropout=0.2, recurrent_dropout=0.2, activation='relu')` – используем для слоя GRU четыре нейрона, `dropout=0.2` – дропаут 20% на входные данные и `recurrent_dropout=0.2` – дропаут 20% на данные, которые возвращаются обратно, обработанные нейроном.
#
# При необходимости можно вернуть еще и промежуточные выходы (скрытые состояния) каждой ячейки на всех итерациях: $h_0, ..., h_n$. Для этого в рекуррентном слое устанавливаем параметр `return_sequences=True`.
#
# Можно дополнительно вернуть еще и внутренние состояния ячеек на всех итерациях, установив парамтр `return_state=True`.
#
# По умолчанию `return_sequences=False` и `return_state=False`.

# %% [markdown] id="ZgoTyr4ENWQa"
# ### Загрузка предобученных Embedding

# %% colab={"base_uri": "https://localhost:8080/"} id="KAShKdNW1IOB" outputId="5a9ca19e-a774-4489-98eb-f7cc9c74a409"
# !wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

# %% colab={"base_uri": "https://localhost:8080/"} id="k9q_8I1e_rFv" outputId="07ae3d9e-7d49-426f-c41d-9f2d622f038b"
# !pip install navec

# %% id="L3jrHxnUwH0C"
from navec import Navec
navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

# %% [markdown] id="1j7ewHxgtPF6"
# ### Подготовка датасета

# %% [markdown] id="9TdRA02pubSB"
# Повторим шаги из прошлого урока:

# %% [markdown] id="-vYnZFaJunfS"
# **Шаг 1. Загрузка базы писателей Русской литературы**

# %% colab={"base_uri": "https://localhost:8080/"} id="BDbQZuPMrNK7" outputId="d2a88f57-42e0-407e-dc31-444affabe917"
data_path = get_file(
    "russian_literature.zip",
    "https://storage.yandexcloud.net/academy.ai/russian_literature.zip"
)

# %% [markdown] id="TU5AzTSbtWOA"
# **Шаг 2. Распаковка датасета**

# %% id="ifmhCktTu4bk"
# Разархивируем датасета во временную папку 'dataset'
# !unzip -qo "{data_path}" -d ./dataset

# %% [markdown] id="ch_tHMbvvD9R"
# **Шаг 3. Формирование датасета**

# %% [markdown] id="PTX194zWvqin"
# Сформируем список всех писателей.

# %% colab={"base_uri": "https://localhost:8080/"} id="FlLPhFu3Jq0Y" outputId="e82f676b-751b-4024-d4e1-f35a53deed2b"
FILE_DIR_POEMS = './dataset/poems'
file_list_poems = os.listdir(FILE_DIR_POEMS)
print("Поэты:")
print(file_list_poems)

# %% colab={"base_uri": "https://localhost:8080/"} id="0-KPPFD3Kqd4" outputId="27bccfe0-9b90-4442-d6fc-ddf7c67c3940"
FILE_DIR_PROSE = './dataset/prose'
file_list_prose = os.listdir(FILE_DIR_PROSE)
print("Прозаики:")
print(file_list_prose)

# %% [markdown] id="dBqkpekULRXr"
# Объединим списки и избавимся от дублей. Для этого с помощью операции сложения (конкатенации) списков объединим списки. Затем преобразуем их в множества (вспомним, что множества содержат только уникальные значения). И после снова преобразуем в список:

# %% colab={"base_uri": "https://localhost:8080/"} id="NIuvY29cLVLU" outputId="d589753d-8e6a-45a8-9d65-a233f3696674"
CLASS_LIST = list(set(file_list_poems + file_list_prose))
print("Общий список писателей:")
print(CLASS_LIST)

# %% id="jhPT6_7UvSMP"
all_texts = {} # Собираем в словарь весь датасет

for author in CLASS_LIST:
    all_texts[author] = '' # Инициализируем пустой строкой новый ключ словаря
    for path in glob.glob('./dataset/prose/{}/*.txt'.format(author)) +  glob.glob('./dataset/poems/{}/*.txt'.format(author)): # Поиск файлов по шаблону
        with open(f'{path}', 'r', errors='ignore') as f: # игнорируем ошибки (например символы из другой кодировки)
            # Загрузка содержимого файла в строку
            text = f.read()

        all_texts[author]  += ' ' + text.replace('\n', ' ') # Заменяем символ перехода на новую строку пробелом

# %% [markdown] id="7UOdeAHAzGC5"
# **Шаг 4. Токенизация**

# %% id="6kzsJdQVyX-Z"
embedding_dim = 300    # размерность векторов эмбединга (300d в имени эмбединга)
max_words = 15000      # Количество слов, рассматриваемых как признаки

# Используется встроенный в Keras токенизатор для разбиения текста и построения частотного словаря
tokenizer = Tokenizer(num_words=max_words,
                      filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
                      lower=True, split=' ', char_level=False)


# Построение частотного словаря по текстам
tokenizer.fit_on_texts(all_texts.values())

# %% [markdown] id="apH9cT02onBg"
# Обратите внимание, что мы увеличили колчество слов `max_words` до 15000. Это стало возможно, так как архитектуры на рекуррентных и одномерных свертках потребляют меньше оперативной памяти, чем полносвязные слои. Что потенциально должно увеличить точность классификации.

# %% [markdown] id="halNnOWDz9Df"
# **Шаг 5. Преобразование текста в последовательность**

# %% [markdown] id="ViH0AZ130GTS"
# Преобразуем текст в последовательности:

# %% id="AgtYOu6qyX6-"
seq_train = tokenizer.texts_to_sequences(all_texts.values())

# %% [markdown] id="WGiGwwtso5EH"
# **Шаг 6. Балансировка датасета**

# %% [markdown] id="a1jBwFwGSJFD"
# Для того, чтобы мы могли наглядно сравнить архитектуры рекурентных, одномерных сверток с полносвязанными слоями (из урока [20.2](https://colab.research.google.com/drive/1KEFUgyBcqGaXGZEU-7MHENn5RH_AIvfH)), мы должны взять аналогичные примеры для обучения, поэтому применим аналогичную балансировку с отсечкой писателей с небольшими текстами.

# %% colab={"base_uri": "https://localhost:8080/"} id="YYZYCwL1eWga" outputId="4ece164c-966c-45e6-f89a-d28da9a71e9c"
# используем генератор цикла для получения длины текстов по каждому автору
total = sum(len(i) for i in seq_train)
print(f'Датасет состоит из {total} слов')

print('Общая выборка по писателям (по словам):')
mean_list = np.array([])
for author in CLASS_LIST:
    cls = CLASS_LIST.index(author)
    print(f'{author} - {len(seq_train[cls])} слов, доля в общей базе: {len(seq_train[cls])/total*100 :.2f}%')
    mean_list = np.append(mean_list, len(seq_train[cls]))

print('Среднее значение слов: ', np.round(mean_list.mean()))
print('Медианное значение слов: ', np.median(mean_list))


median = int(np.median(mean_list)) # Зафиксировали медианное значение
CLASS_LIST_BALANCE = [] # Сбалансированный набор меток
seq_train_balance = []
for author in CLASS_LIST:
    cls = CLASS_LIST.index(author)
    if len(seq_train[cls]) > median * 0.6:
      seq_train_balance.append(seq_train[cls][:median])
      CLASS_LIST_BALANCE.append(author)

# %% colab={"base_uri": "https://localhost:8080/"} id="adjLWNQCebVT" outputId="38a5c6f9-517a-4081-8fc3-0c0d93ab506f"
total = sum(len(i) for i in seq_train_balance)

print('Сбалансированная выборка по писателям (по словам):')
mean_list_balance = np.array([])
for author in CLASS_LIST_BALANCE:
    cls = CLASS_LIST_BALANCE.index(author)
    print(f'{author} - {len(seq_train_balance[cls])} слов, доля в общей базе: {len(seq_train_balance[cls])/total*100 :.2f}%')
    mean_list_balance = np.append(mean_list_balance, len(seq_train_balance[cls]))

print('Среднее значение слов: ', np.round(mean_list_balance.mean()))
print('Медианное значение слов: ', np.median(mean_list_balance))

# %% colab={"base_uri": "https://localhost:8080/", "height": 451} id="cCuJYCJaeqUT" outputId="a95b7a56-a4db-4b6b-9933-1353f64b79f7"
fig, ax = plt.subplots()
ax.pie([len(i) for i in seq_train_balance],  # формируем список значений как длина символов текста каждого автора
       labels=CLASS_LIST_BALANCE,            # список меток
       pctdistance=1.2,                      # дистанция размещения % (1 - граница окружности)
       labeldistance=1.4,                    # размещение меток (1 - граница окружности)
       autopct='%1.2f%%'                     # формат для % (2 знака после запятой)
      )
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="lAPccR_JBx1Q" outputId="f2c8f5a8-77fd-4cce-b766-61d6948aab89"
# используем генератор цикла для получения длины текстов по каждому автору
total = sum(len(i) for i in seq_train_balance)
print(f'Датасет состоит из {total} слов')

# %% colab={"base_uri": "https://localhost:8080/"} id="R526W9_6Bqe1" outputId="7dee8c41-e3ee-4356-e1e7-9b06781b9230"
print('Общая выборка по писателям (по словам):')
mean_list = np.array([])
for author in CLASS_LIST_BALANCE:
    cls = CLASS_LIST_BALANCE.index(author)
    print(f'{author} - {len(seq_train_balance[cls])} слов, доля в общей базе: {len(seq_train_balance[cls])/total*100 :.2f}%')
    mean_list = np.append(mean_list, len(seq_train_balance[cls]))

print('Среднее значение слов: ', np.round(mean_list.mean()))
print('Медианное значение слов: ', np.median(mean_list))

# %% colab={"base_uri": "https://localhost:8080/", "height": 451} id="jXlqI1HNDGDE" outputId="eaa8a787-3838-4e8c-9659-f563f5208a39"
fig, ax = plt.subplots()
ax.pie([len(i) for i in seq_train_balance], # формируем список значений как длина символов текста каждого автора
       labels=CLASS_LIST_BALANCE,                    # список меток
       pctdistance=1.2,                      # дистанция размещения % (1 - граница окружности)
       labeldistance=1.4,                    # размещение меток (1 - граница окружности)
       autopct='%1.2f%%'                     # формат для % (2 знака после запятой)
      )
plt.show()

# %% [markdown] id="Gg0ce5LLAdje"
# **Шаг 7. Нарезка примеров из текста методом скользящего окна**

# %% id="j5IDCSoID53M"
WIN_SIZE = 1000   # Ширина окна в токенах
WIN_STEP = 100    # Шаг окна в токенах

# Функция разбиения последовательности на отрезки скользящим окном
# Последовательность разбивается на части до последнего полного окна
# Параметры:
# sequence - последовательность токенов
# win_size - размер окна
# step - шаг окна
def seq_split(sequence, win_size, step):
    # Делим строку на отрезки с помощью генератора цикла
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, step)]

def seq_vectorize(
    seq_list,   # Последовательность
    test_split, # Доля на тестовую выборку
    class_list, # Список классов
    win_size,   # Ширина скользящего окна
    step        # Шаг скользящего окна
):

    # Списки для результирующих данных
    x_train, y_train, x_test, y_test =  [], [], [], []

    # Пробежимся по всем классам:
    for class_item in class_list:
        # Получим индекс класса
        cls = class_list.index(class_item)

        # Пороговое значение индекса для разбивки на тестовую и обучающую выборки
        gate_split = int(len(seq_list[cls]) * (1-test_split))

        # Разбиваем последовательность токенов класса на отрезки
        vectors_train = seq_split(seq_list[cls][:gate_split], win_size, step) # последовательность до порога попадет в обучающую выборку
        vectors_test = seq_split(seq_list[cls][gate_split:], win_size, step)  # последовательность после порога попадет в тестовую выборку

        # Добавляем отрезки в выборку
        x_train += vectors_train
        x_test += vectors_test

        # Для всех отрезков класса добавляем метки класса в виде one-hot-encoding
        # Каждую метку берем len(vectors) раз, так она одинакова для всех выборок одного класса
        y_train += [to_categorical(cls, len(class_list))] * len(vectors_train)
        y_test += [to_categorical(cls, len(class_list))] * len(vectors_test)

    # Возвращаем результатов как numpy-массивов
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


# %% id="DeULeastEBng"
x_train, y_train, x_test, y_test = seq_vectorize(seq_train_balance, 0.1, CLASS_LIST_BALANCE, WIN_SIZE, WIN_STEP)

# %% colab={"base_uri": "https://localhost:8080/"} id="AD-WovkzFBK4" outputId="b311148e-8f56-49d6-f23f-8cb964bb948f"
print(f'Форма входных данных для обучающей выборки: {x_train.shape}')
print(f'Форма выходных данных (меток) для обучающей выборки: {y_train.shape}')

# %% colab={"base_uri": "https://localhost:8080/"} id="YHK3dF-XFDKu" outputId="3490e3f4-a068-41e4-b388-653543861ac6"
print(f'Форма входных данных для тестовой выборки: {x_test.shape}')
print(f'Форма выходных данных (меток) для тестовой выборки: {y_test.shape}')


# %% [markdown] id="bAppo-vS2qYw"
# **Шаг 8. Определим вспомагательные функции**

# %% id="_mXa5HgxFXjC"
# Вывод графиков точности и ошибки
def show_plot(history, title=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('График процесса обучения модели: '+title)
    ax1.plot(history.history['accuracy'],
               label='График точности на обучающей выборке')
    ax1.plot(history.history['val_accuracy'],
               label='График точности на проверочной выборке')
    ax1.xaxis.get_major_locator().set_params(integer=True) # На оси х показываем целые числа
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('График точности')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='Ошибка на обучающей выборке')
    ax2.plot(history.history['val_loss'],
               label='Ошибка на проверочной выборке')
    ax2.xaxis.get_major_locator().set_params(integer=True) # На оси х показываем целые числа
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    plt.show()

# Функция вывода предсказанных значений
def show_confusion_matrix(y_true, y_pred, class_labels):
    # Матрица ошибок
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    # Округление значений матрицы ошибок
    cm = np.around(cm, 3)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f'Матрица ошибок', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()  # Убираем ненужную цветовую шкалу
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    fig.autofmt_xdate(rotation=45)          # Наклон меток горизонтальной оси
    plt.show()


    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    print('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))


# %% [markdown] id="KHFPFwlHQ1SR"
# Также определим функцию загрузки в модель весов Наташи для Embedding:

# %% id="l3Knq7gUQ1oc"
def loadEmbedding(model):
    word_index = tokenizer.word_index
    embeddings_index = navec

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    return model


# %% [markdown] id="9RYCNWBhiYW7"
# ### Сравнение архитектур сети

# %% [markdown] id="RWv1HckriVQc"
# Так как мы используем предварительно сформированные векторные представления, то это накладывает ограничения на размерность входного слоя `Embedding` нашей модели (смотрим по имени файла navec_hudlit_v1_12B_500K_**300d**_100q.tar). Поэтому во всех наших экспериментах мы будем использовать один и тот же обученный входной слой `Embedding`.

# %% [markdown] id="Ik4cj2FtPXVw"
# #### **Embedding(Natasha) + SimpleRNN(10) + Dropout(0.3)**

# %% id="1FLZplc4QZ-b"
# Создание последовательной модели нейросети
model_SimpleRNN_1 = Sequential()

model_SimpleRNN_1.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))

# Слой регуляризации, "выключает" 1D карты объектов из эмбеддинг-векторов
model_SimpleRNN_1.add(SpatialDropout1D(0.3))
# Слой нормализации данных
model_SimpleRNN_1.add(BatchNormalization())
# Рекуррентный слой
model_SimpleRNN_1.add(SimpleRNN(10))
# Слой регуляризации Dropout для отдельных нейронов
model_SimpleRNN_1.add(Dropout(0.3))
# Выходной слой классификатора
model_SimpleRNN_1.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="LvPSibrmSWYk"
model_SimpleRNN_1 = loadEmbedding(model_SimpleRNN_1)

# %% colab={"base_uri": "https://localhost:8080/"} id="fTDGwqvqKTlw" outputId="6f23372b-4d60-4e0a-d743-32500d204e91"
model_SimpleRNN_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_SimpleRNN_1.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))


# %% colab={"base_uri": "https://localhost:8080/", "height": 400} id="tXhRPFU_IdIa" outputId="26e329a7-8593-4ca7-8be8-07a9afa13568"
show_plot(history, 'Embedding(Natasha) + SimpleRNN(10) + Dropout(0.3)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="hXB8iTN2IdvE" outputId="3c1be2de-c3d2-4c97-d4ac-11deeb24ab96"
y_pred = model_SimpleRNN_1.predict(x_test)
show_confusion_matrix(y_test, y_pred, CLASS_LIST_BALANCE)

# %% [markdown] id="sTZGUmSiT6FO"
# #### **Embedding(Natasha) + SimpleRNN(5) + Dropout(0.2)**

# %% id="3AW4rqsuUVNR"
# Создание последовательной модели нейросети
model_SimpleRNN_2 = Sequential()

model_SimpleRNN_2.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))

# Слой регуляризации, "выключает" 1D карты объектов из эмбеддинг-векторов
model_SimpleRNN_2.add(SpatialDropout1D(0.2))
# Слой нормализации данных
model_SimpleRNN_2.add(BatchNormalization())
# Рекуррентный слой
model_SimpleRNN_2.add(SimpleRNN(5))
# Слой регуляризации Dropout для отдельных нейронов
model_SimpleRNN_2.add(Dropout(0.2))
# Выходной слой классификатора
model_SimpleRNN_2.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="53vi2wnDUgHg"
model_SimpleRNN_2 = loadEmbedding(model_SimpleRNN_2)

# %% colab={"base_uri": "https://localhost:8080/"} id="EVRFTgSRUyCX" outputId="faef14a9-6b7a-4282-81fa-fce91edc37df"
model_SimpleRNN_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model_SimpleRNN_2.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 293} id="cmjow47GUyh5" outputId="b7dd3244-e229-4053-d267-4ef1d0e0bf79"
show_plot(history2, 'Embedding(Natasha) + SimpleRNN(40) + Dropout(0.3)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="HK3MjrZPVRiA" outputId="08da5de5-17a0-43e2-c947-09bd8f77fbf7"
y_pred2 = model_SimpleRNN_2.predict(x_test)
show_confusion_matrix(y_test, y_pred2, CLASS_LIST_BALANCE)

# %% [markdown] id="GxMHGdDWUu3D"
# #### **Embedding(Natasha) + GRU(10) + Dropout(0.2)**

# %% [markdown] id="BYbV58cXITpO"
# Слой GRU в Keras с параметрами регуляризации не считается на cuda ядрах GPU процессора. При запуске на GPU увидите предупреждение. Поэтому расчеты будут производиться долго и придется запастись терпением.

# %% id="gtCl2tu2YiHu"
model_GRU_3 = Sequential()
model_GRU_3.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))
model_GRU_3.add(SpatialDropout1D(0.2))
model_GRU_3.add(BatchNormalization())
# Рекуррентный слой GRU
model_GRU_3.add(GRU(10, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
model_GRU_3.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="VLaWBS8cZdE0"
model_GRU_3 = loadEmbedding(model_GRU_3)

# %% colab={"base_uri": "https://localhost:8080/"} id="YNPBsTd2Zf-J" outputId="6f22283e-259f-46c3-c83c-c9a6fe27d182"
model_GRU_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history3 = model_GRU_3.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 395} id="QdSrJMcfZg6-" outputId="4341062e-20ca-4f3d-c860-d997245f7c4e"
show_plot(history3, 'Embedding(Natasha) + GRU(10) + Dropout(0.2)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="PuF07KGRZkDA" outputId="38a086a9-e98b-48ea-b35e-03941dd89c4b"
y_pred3 = model_GRU_3.predict(x_test)
show_confusion_matrix(y_test, y_pred3, CLASS_LIST_BALANCE)

# %% [markdown] id="VWYmu4eTbUIC"
# #### **Embedding(Natasha) + GRU(40) + Dropout(0.2)**

# %% id="UpglWSxebYMj"
model_GRU_4 = Sequential()
model_GRU_4.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))
model_GRU_4.add(SpatialDropout1D(0.2))
model_GRU_4.add(BatchNormalization())
# Рекуррентный слой GRU
model_GRU_4.add(GRU(40, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
model_GRU_4.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="B_nSPCOCbc-p"
model_GRU_4 = loadEmbedding(model_GRU_4)

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 408} id="tinAn7GFbffT" outputId="d18aaafe-0dad-4ff8-d734-f9fbf9e341bf"
model_GRU_4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history4 = model_GRU_4.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"background_save": true} id="kB6wWn4qbjSz" outputId="60d6b00a-a797-4410-ac30-7743f03bbe5a"
show_plot(history4, 'Embedding(Natasha) + GRU(40) + Dropout(0.2)')

# %% colab={"background_save": true} id="sYKYzRxrbmsq" outputId="2ddaff8b-2464-456c-fba8-3a3978045d3a"
y_pred4 = model_GRU_4.predict(x_test)
show_confusion_matrix(y_test, y_pred4, CLASS_LIST_BALANCE)

# %% [markdown] id="YLH2PPQ0uG6m"
# #### **Embedding(Natasha) + LSTM(20) + Dropout(0.2)**

# %% id="PgrVnsL1uDNV"
model_LSTM_5 = Sequential()
model_LSTM_5.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))
model_LSTM_5.add(SpatialDropout1D(0.2))
model_LSTM_5.add(BatchNormalization())

# Рекуррентный слой LSTM
model_LSTM_5.add(LSTM(20))
model_LSTM_5.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="POOgToF_uk3P"
model_LSTM_5 = loadEmbedding(model_LSTM_5)

# %% colab={"base_uri": "https://localhost:8080/"} id="PFu1Y1yGunR6" outputId="0e9f5ca6-99d8-4738-b94b-0ed35a755ac8"
model_LSTM_5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history5 = model_LSTM_5.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 402} id="g-mzGAwYuq3N" outputId="74048bdc-8ee0-4329-d396-3c4aaa60ba25"
show_plot(history5, 'Embedding(Natasha) + LSTM(20) + Dropout(0.2)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="DH1oxQg2uqyj" outputId="050694a3-f76e-42a9-c93e-98372992e7da"
y_pred5 = model_LSTM_5.predict(x_test)
show_confusion_matrix(y_test, y_pred5, CLASS_LIST_BALANCE)

# %% [markdown] id="rFamFFTAvPuT"
# #### **Embedding(Natasha) + LSTM(100) + Dropout(0.3)**

# %% id="weCdQaoBvPuU"
model_LSTM_6 = Sequential()
model_LSTM_6.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))
model_LSTM_6.add(SpatialDropout1D(0.3))
model_LSTM_6.add(BatchNormalization())

# Рекуррентный слой LSTM
model_LSTM_6.add(LSTM(100))
model_LSTM_6.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="W2sGjrq5vPuU"
model_LSTM_6 = loadEmbedding(model_LSTM_6)

# %% colab={"base_uri": "https://localhost:8080/"} id="lQ9pgMbwvPuU" outputId="da070a06-3fb0-4ad7-b528-681c75b5185f"
model_LSTM_6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history6 = model_LSTM_6.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 497} id="Ii4jOw3wvPuU" outputId="11688f01-2a73-4237-faa1-fa9ac16336d3"
show_plot(history6, 'Embedding(Natasha) + LSTM(100) + Dropout(0.3)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="9lg2eY0LvPuU" outputId="c5cc78d1-c49b-4341-95ca-cf97048c7769"
y_pred6 = model_LSTM_6.predict(x_test)
show_confusion_matrix(y_test, y_pred6, CLASS_LIST_BALANCE)

# %% [markdown] id="c5RfIC5loJ_E"
# #### **Embedding(Natasha) + BLSTM(8)x2 + GRU(16)x2 + Dropout(0.3) + Dense(100)**

# %% [markdown] id="7dwIyZ-nzYoY"
# `Bidirectional(LSTM(8, return_sequences=True))` – этот слой
# активизирует двунаправленную сеть **LSTM**;
#
# `GRU(16, return_sequences=True, reset_after=True)` – параметр `reset_after=True` означает сброс данных.

# %% id="H6jCCMhPkseq"
model_MIX = Sequential()
model_MIX.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))
model_MIX.add(SpatialDropout1D(0.3))
model_MIX.add(BatchNormalization())

# Два двунаправленных рекуррентных слоя LSTM
model_MIX.add(Bidirectional(LSTM(8, return_sequences=True)))
model_MIX.add(Bidirectional(LSTM(8, return_sequences=True)))
model_MIX.add(Dropout(0.3))
model_MIX.add(BatchNormalization())

# Два рекуррентных слоя GRU
model_MIX.add(GRU(16, return_sequences=True, reset_after=True))
model_MIX.add(GRU(16, reset_after=True))
model_MIX.add(Dropout(0.3))
model_MIX.add(BatchNormalization())

# Дополнительный полносвязный слой
model_MIX.add(Dense(100, activation='relu'))
model_MIX.add(Dropout(0.3))
model_MIX.add(BatchNormalization())
model_MIX.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="pBn0gisVp07u"
model_MIX = loadEmbedding(model_MIX)

# %% colab={"base_uri": "https://localhost:8080/"} id="m5kGicpop4BX" outputId="6167c480-124c-42a7-908a-4ef8cc419f5d"
model_MIX.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_mix = model_MIX.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 402} id="HlsaGYsVp6aw" outputId="a01026fa-9ee3-41ed-9490-ac8ad97da4ab"
show_plot(history_mix, 'Embedding(Natasha) + BLSTM(8)x2 + GRU(16)x2 + Dropout(0.3) + Dense(100)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 899} id="8X9z3bSWp81A" outputId="851f3d68-90dd-4622-d64f-e51af396006e"
y_pred_mix = model_MIX.predict(x_test)
show_confusion_matrix(y_test, y_pred_mix, CLASS_LIST_BALANCE)

# %% [markdown] id="jXKOhmD35VZb"
# #### **Embedding(Natasha) + Conv1D(20)x2 + Dropout(0.2)**

# %% id="MfXu7yDh5Vy6"
model_Conv1D = Sequential()
model_Conv1D.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))
model_Conv1D.add(SpatialDropout1D(0.2))
model_Conv1D.add(BatchNormalization())
# Два слоя одномерной свертки Conv1D
model_Conv1D.add(Conv1D(20, 5, activation='relu', padding='same'))
model_Conv1D.add(Conv1D(20, 5, activation='relu'))
# Слой подвыборки/пулинга с функцией максимума
model_Conv1D.add(MaxPooling1D(2))
model_Conv1D.add(Dropout(0.2))
# Слой пакетной нормализации
model_Conv1D.add(BatchNormalization())
# Слой выравнивания в вектор
model_Conv1D.add(Flatten())
model_Conv1D.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="GBWEbyC57D8A"
model_Conv1D = loadEmbedding(model_Conv1D)

# %% colab={"base_uri": "https://localhost:8080/"} id="235x0NVq7MfZ" outputId="ad34b251-8e6b-461d-c1b9-cadaa46c7738"
model_Conv1D.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_1D = model_Conv1D.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 287} id="8Ma02eMM7RUa" outputId="e9cb6543-82d0-4004-e4c4-8deeee769770"
show_plot(history_1D, 'Embedding(Natasha) + Conv1D(20)x2 + Dropout(0.2)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 877} id="q81n6rBd7gQr" outputId="3e95728c-edad-4f75-f4cb-0969d07a057a"
y_pred_1d = model_Conv1D.predict(x_test)
show_confusion_matrix(y_test, y_pred_1d, CLASS_LIST_BALANCE)

# %% [markdown] id="7riJKlyc5mO3"
# #### **Embedding(Natasha) + Conv1D(20)x3 + LSTM(4)x2 + Dropout(0.2) + Dense(100)**

# %% id="YzYoqAol5mun"
model_Conv_LSTM = Sequential()
model_Conv_LSTM.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE))
model_Conv_LSTM.add(SpatialDropout1D(0.2))
# Рекуррентный слой LSTM
model_Conv_LSTM.add(LSTM(4, return_sequences=1))
# Полносвязный слой
model_Conv_LSTM.add(Dense(100, activation='relu'))
# Сверточный слой
model_Conv_LSTM.add(Conv1D(20, 5, activation='relu'))
# Рекуррентный слой LSTM
model_Conv_LSTM.add(LSTM(4, return_sequences=1))
# Слой регуляризации Dropout
model_Conv_LSTM.add(Dropout(0.2))
# Слой пакетной нормализации
model_Conv_LSTM.add(BatchNormalization())
# Два сверточных слоя
model_Conv_LSTM.add(Conv1D(20, 5, activation='relu'))
model_Conv_LSTM.add(Conv1D(20, 5, activation='relu'))
# Слой подвыборки/пулинга с функцией максимума
model_Conv_LSTM.add(MaxPooling1D(2))
model_Conv_LSTM.add(Dropout(0.2))
model_Conv_LSTM.add(BatchNormalization())
# Слой выравнивания в вектор
model_Conv_LSTM.add(Flatten())
model_Conv_LSTM.add(Dense(len(CLASS_LIST_BALANCE), activation='softmax'))

# %% id="aR2OR8LH79W3"
model_Conv_LSTM = loadEmbedding(model_Conv_LSTM)

# %% colab={"base_uri": "https://localhost:8080/"} id="1ZKryvBN79W9" outputId="138d01f2-df18-4e1a-c175-b2b42a62fc05"
model_Conv_LSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_1D_LSTM = model_Conv_LSTM.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 287} id="eysNPJan79W9" outputId="88f864d4-dc37-45ca-b51e-2ab15ca5e4d8"
show_plot(history_1D_LSTM, 'Embedding(Natasha) + Conv1D(20)x3 + LSTM(4)x2 + Dropout(0.2) + Dense(100)')

# %% colab={"base_uri": "https://localhost:8080/", "height": 877} id="sHT3Sm9b79W-" outputId="84f4f6f2-15ae-4000-9778-1000aea3df00"
y_pred_lstm = model_Conv_LSTM.predict(x_test)
show_confusion_matrix(y_test, y_pred_lstm, CLASS_LIST_BALANCE)

# %% [markdown] id="Q9SXo-LpL5Ou"
# #### Итоги

# %% [markdown] id="HNCDdUSQL9Qb"
# В данном эксперименте мы использовали одни и те же данные в качестве проверочных и тестовых данных. Мы это сделали нарочно, чтобы данные в матрице ошибки характеризовали наши итоговые данные на проверочной выборке. По ним мы и сравним наши архитектуры. А также для того, чтобы в домашней работе вы аккуратно и последовательно разделили выборки и сделали все по правилам: обучили на обучающей выборке, контролировали ход обучения по проверочной выборке, а итоговую матрицу ошибок построили по тестовой выборке.
#
# Выпишем результаты средней точности распознования по матрице ошибок:
# 1. Embedding(Natasha) + SimpleRNN(10) + Dropout(0.3) - **19%**
# 2. Embedding(Natasha) + SimpleRNN(5) + Dropout(0.2)  - **20%**
# 3. Embedding(Natasha) + GRU(10) + Dropout(0.2) - **75%**
# 4. Embedding(Natasha) + GRU(40) + Dropout(0.2) - **86%**
# 5. Embedding(Natasha) + LSTM(20) + Dropout(0.2) - **83%**
# 6. Embedding(Natasha) + LSTM(100) + Dropout(0.3) - **83%**
# 7. Embedding(Natasha) + BLSTM(8)x2 + GRU(16)x2 + Dropout(0.3) + Dense(100) - **89%**
# 8. Embedding(Natasha) + Conv1D(20)x2 + Dropout(0.2) - **76%**
# 9. Embedding(Natasha) + Conv1D(20)x3 + LSTM(4)x2 + Dropout(0.2) + Dense(100) - **77%**
#
# Как и следовало ожидать результаты на простых SimpleRNN (1 и 2 эксперименты) получились неудовлетворительными, поэтому по возможности не используйте их. Причина такого "скромного" результата заключается в исчезающем градиенте, о котором мы говорили ранее.
#
# Вполне ожидаемо, что в нашем исследовании лучшей оказалась архитектура (7), использующая двухнаправленные LSTM сети в комбинации с GRU и полносвязанными сетями. Можете ее смело применять в своих проектах. Данная комбинация слоев была проверена на большом числе датасетов различных задач классификации текстов.  
#
# Что неожиданно, но простая сверточная сеть (8 эксперимент), показала себя на уровне GRU (3 эксперимент)!
#
# В целом по GRU, LSTM и одномерным сверткам результаты получились похожими и зависимыми от числа нейронов в слое.
#
# Также вы можете самостоятельно проверить, что если мы не будем загружать в Эмбединги предварительно обученные веса из проекта Natasha, то потеряем по точности более 10%.

# %% [markdown] id="bZdzsNwfDar2"
# А теперь пора выполнить [домашнюю работу](https://colab.research.google.com/drive/1NMDG3ZeGgyHm0ei0DOC63PBb9rW6oN3O) с использованием наиболее понравившейся вам архитектурой, либо вы можете придумать свою.
