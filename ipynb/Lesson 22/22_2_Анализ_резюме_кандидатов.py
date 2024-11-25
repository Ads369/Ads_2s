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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/22_2_%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7_%D1%80%D0%B5%D0%B7%D1%8E%D0%BC%D0%B5_%D0%BA%D0%B0%D0%BD%D0%B4%D0%B8%D0%B4%D0%B0%D1%82%D0%BE%D0%B2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="Oh5y9k76hVn5"
#  **Навигация по уроку**
#
# 1. [Решение задач регрессии с помощью НС](https://colab.research.google.com/drive/1GbXbqPbC4A2NVJEj-5pOOySN3gyalBQN)
# 2. Анализ резюме кандидатов
# 3. [Домашняя работа](https://colab.research.google.com/drive/1iPTkGZ_AEUpl5l6DR__J021gHR61RRfQ)

# %% [markdown] id="fi7HrzSZa_nG"
# В данном практическом Блокноте мы обучим НС на наборе данных [Резюме из ЕЦП «Работа в России»](https://opendata.trudvsem.ru/csv/cv.csv) предсказывать ожидаемую зарплату соискателя по его резюме. Датасет можно получить по [ссылке](https://opendata.trudvsem.ru/csv/cv.csv) из официального источника. Однако, он периодически обновляется, имеет размер более 20Гб и может содержать ошибки, что усложняет его загрузку и обработку. Для обучающего примера, он слишком велик. Поэтому предлагается использовать его обрезанную до 100 тыс резюме [копию](https://storage.yandexcloud.net/academy.ai/cv_100000.csv) не превышающую 230Мб с исправленными ошибками.
#
# Как всегда мы начинаем работу с импорта необходимых библиотек.

# %% [markdown] id="7Qo2vGNbbAFb"
# ## Импорт библиотек

# %% id="_pVP7VJubG8f"
# Для работы с массивами данных
import numpy as np

# Для работы с табличными данными
import pandas as pd

# Библиотека утилит
from keras import utils

# Для работы с моделями
from keras.models import Sequential, Model

# Слои
from keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation, Input, concatenate
from keras.layers import SimpleRNN, GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D

# Оптимизаторы
from keras.optimizers import Adam, Adadelta, SGD, Adagrad, RMSprop

# Токенизатор
from tensorflow.keras.preprocessing.text import Tokenizer

# Нормализация данных
from sklearn.preprocessing import StandardScaler

# Регулярные выражения
import re

# Для работы с графиками
import matplotlib.pyplot as plt

# Метрики для расчета ошибок
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Для преобразования строки в json формат
import json

# %matplotlib inline

# %% [markdown] id="JXP1J32Ub-FK"
# ## Загрузка данных

# %% [markdown] id="K9VwbXqmimwl"
# Далее мы загружаем облегченную версию датасета. Порядок и набор полей сохранен как в оригинальном датасете.

# %% id="JgEFjaeyKGjf" colab={"base_uri": "https://localhost:8080/"} outputId="2aca5c70-7f39-4fcc-85c7-465cae811ba7"
# !wget https://storage.yandexcloud.net/academy.ai/cv_100000.csv

# %% colab={"base_uri": "https://localhost:8080/"} id="3aW4tAhacRK-" outputId="c67e1a14-aa8d-4f76-af6a-5ad64a108ffe"
# Чтение файла базы данных
df = pd.read_csv('cv_100000.csv', delimiter='|', on_bad_lines='skip', low_memory=False, index_col=0)

# Вывод количества резюме и числа признаков
print('Форма данных: ', df.shape)


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="gc_bXDhhzqy6" outputId="5cfc41fb-75f6-41fc-be95-3158d5ba29b8"
# Вывод первых 10 значений датасета
df.head(10)

# %% [markdown] id="oR673xC1Oj3E"
# Удалим ненужные столбцы, чтобы не перегружать оперативную память при работе с данными:

# %% id="LWBtQD_sOjSC"
df.drop(['id', 'candidateId', 'stateRegionCode', 'locality', 'birthday', 'gender', 'dateCreate', 'dateModify',
         'publishedDate', 'academicDegree', 'worldskills', 'worldskillsInspectionStatus', 'abilympicsInspectionStatus',
         'abilympicsParticipation', 'volunteersInspectionStatus', 'volunteersParticipation', 'driveLicenses', 'professionsList',
         'otherCertificates', 'narkCertificate', 'narkInspectionStatus', 'codeExternalSystem', 'country', 'additionalEducationList',
         'hardSkills', 'softSkills', 'retrainingCapability', 'businessTrip', 'languageKnowledge', 'relocation', 'innerInfo'

         ], inplace=True, axis=1)

# %% [markdown] id="ZS10gGv1gYC_"
# ## Обработка данных

# %% [markdown] id="wAFTXhiCfnhG"
# Выведем произвольный пример и внимательно посмотрим на данные:

# %% colab={"base_uri": "https://localhost:8080/"} id="kHTZOEW4foAa" outputId="b3aa813d-7898-4de1-fbdb-8f3a5275d2b9"
# Пример данных

n = 3                                     # Индекс в таблице резюме
for i in range(len(df.values[n])):        # Вывод значения каждого столбца
    print('{:>2} {:>30}  {}'.format(i, df.columns[i], df.values[n][i]))

# %% [markdown] id="5UqW7tPZ7u8U"
# Для наглядности представим в виде таблицы доступное на [портале](https://trudvsem.ru/opendata/datasets) описание полей атрибутов данного датасета в формате json.

# %% [markdown] id="gjPgz0sD06QP"
# <table>
# <tr><th>Код</th><th>Имя</th><th>Тип</th><th>Пример</th></tr>
# <tr><td>id</td><td>Идентификатор резюме на Портале</td><td>timeuuid</td><td>08b23f30-3eb7-11ed-be3b-f5407cdefa62</td></tr>
# <tr><td>stateRegionCode</td><td>Код федерального региона</td><td>varchar</td><td>9900000000000</td></tr>
# <tr><td>locality</td><td>Код населенного пункта</td><td>varchar</td><td>4000000100000</td></tr>
# <tr><td>localityName</td><td>Наименование населенного пункта</td><td>varchar</td><td>Калужская область, г. Калуга</td></tr>
# <tr><td>birthday</td><td>Дата рождения</td><td>date</td><td>2018-11-30</td></tr>
# <tr><td>age</td><td>Возраст</td><td>int</td><td>25</td></tr>
# <tr><td>gender</td><td>Пол</td><td>varchar</td><td>Мужской</td></tr>
# <tr><td>positionName</td><td>Наименование резюме</td><td>varchar</td><td>Воспитатель</td></tr>
# <tr><td>dateCreate</td><td>Дата создания</td><td>date</td><td>2022-12-01T14:54:42+0000</td></tr>
# <tr><td>dateModify</td><td>Дата изменения</td><td>date</td><td>2022-12-01T14:54:42+0000</td></tr>
# <tr><td>publishedDate</td><td>Дата публикации</td><td>date</td><td>2022-12-01T14:54:42+0000</td></tr>
# <tr><td>academicDegree</td><td>Ученая степень</td><td>varchar</td><td>Кандидат наук</td></tr>
# <tr><td>worldskills</td><td>Информация о компетенции Worldskills</td><td> - </td><td> - </td></tr>
# <tr><td>Worldskills</td><td>Компетенции Worldskills</td><td> - </td><td> - </td></tr>
# <tr><td>type</td><td>Тип</td><td>varchar</td><td>Participation</td></tr>
# <tr><td>russianName</td><td>Наименование компетенции на русском</td><td>varchar</td><td>Дизайн</td></tr>
# <tr><td>internationalName</td><td>Наименование компетенции на английском</td><td>varchar</td><td>Design</td></tr>
# <tr><td>isInternational</td><td>Является компетенция международной (WSR)</td><td>boolean</td><td>true</td></tr>
# <tr><td>skillAbbreviation</td><td>Номер компетенции</td><td>varchar</td><td>T29</td></tr>
# <tr><td>worldskillsInspectionStatus</td><td>Статус подтверждения участия в чемпионате WordSkills</td><td>varchar</td><td>Участие в чемпионате WordSkills подтверждено</td></tr>
# <tr><td>сompetitions</td><td>Компетенции</td><td> - </td><td> - </td></tr>
# <tr><td>abilympics</td><td>Международное движение Абилимпикс</td><td> - </td><td> - </td></tr>
# <tr><td>abilympicsParticipation</td><td>Признак участия в движении Абилимпикc</td><td>boolean</td><td>true</td></tr>
# <tr><td>abilympicsInspectionStatus</td><td>Статус подтверждения участия в движении Абилимпикc</td><td>varchar</td><td>Участие в движении Абилимпикc подтверждено</td></tr>
# <tr><td>driveLicenses</td><td>Водительские права</td><td>varchar</td><td>A</td></tr>
# <tr><td>volunteers</td><td>Участие в движении DOBRO.RU</td><td> - </td><td> - </td></tr>
# <tr><td>volunteers</td><td>Движение DOBRO.RU</td><td> - </td><td> - </td></tr>
# <tr><td>volunteersParticipation</td><td>Признак участия в движении DOBRO.RU</td><td>boolean</td><td>true</td></tr>
# <tr><td>volunteersInspectionStatus</td><td>Статус участия в движении DOBRO.RU</td><td>varchar</td><td>Участие в движении DOBRO.RU подтверждено</td></tr>
# <tr><td>nark</td><td>Проверка сертификата НАРК</td><td> - </td><td> - </td></tr>
# <tr><td>narkCertificate</td><td>Флаг проверки сертификата национальным агентством развития квалификаций</td><td>boolean</td><td>true</td></tr>
# <tr><td>narkInspectionStatus</td><td>Статус проверки квалификации НАРК</td><td>varchar</td><td>Получено свидетельство о независимой оценке квалификации</td></tr>
# <tr><td>experience</td><td>Опыт работы</td><td>int</td><td>5</td></tr>
# <tr><td>workExperienceList</td><td>Информация об опыте работы</td><td> - </td><td> - </td></tr>
# <tr><td>workExperience</td><td>Опыт работы</td><td> - </td><td> - </td></tr>
# <tr><td>dateFrom</td><td>Дата начала работы</td><td>date</td><td>2013-09-01</td></tr>
# <tr><td>dateTo</td><td>Дата увольнения</td><td>date</td><td>2017-12-01</td></tr>
# <tr><td>companyName</td><td>Название компании</td><td>varchar</td><td>МБДОУ  г.Калуги</td></tr>
# <tr><td>jobTitle</td><td>Название должности</td><td>varchar</td><td>Воспитатель</td></tr>
# <tr><td>achievements</td><td>Достижения</td><td>varchar</td><td>Улучшение производительности отдела на 15%</td></tr>
# <tr><td>type</td><td>Тип записи</td><td>varchar</td><td>WorkExperience</td></tr>
# <tr><td>demands</td><td>Должностные обязанности</td><td>varchar</td><td>Присмотр и уход за детьми дошкольного учреждения, выполнение режима дня и сетки занятий</td></tr>
# <tr><td>relevant</td><td>Опыт является релевантным</td><td>boolean</td><td>true</td></tr>
# <tr><td>professionList</td><td>Профессиональная сфера</td><td> - </td><td> - </td></tr>
# <tr><td>type</td><td>Тип</td><td>varchar</td><td>CandidateProfession</td></tr>
# <tr><td>codeProfessionalSphere</td><td>Код сферы</td><td>varchar</td><td>Finances</td></tr>
# <tr><td>codeProfession</td><td>Код профессии</td><td>varchar</td><td>233699</td></tr>
# <tr><td>educationList</td><td>Информация об образовании</td><td> - </td><td> - </td></tr>
# <tr><td>education</td><td>Образование</td><td> - </td><td> - </td></tr>
# <tr><td>type</td><td>Вид образования</td><td>varchar</td><td>Education</td></tr>
# <tr><td>level</td><td>Уровень образования</td><td>varchar</td><td>Среднее</td></tr>
# <tr><td>diplomaName</td><td>Наименование диплома</td><td>varchar</td><td>Совершенствование контрольной работы в Межрайонной инспекции ФНС России №7 по Курганской области</td></tr>
# <tr><td>instituteName</td><td>Название учебного заведения</td><td>varchar</td><td>БГТУ</td></tr>
# <tr><td>graduateYear</td><td>Год окончания</td><td>data</td><td>2017-12-01</td></tr>
# <tr><td>faculty</td><td>Факультет</td><td>varchar</td><td>Факультет физической культуры</td></tr>
# <tr><td>specialty</td><td>Специальность</td><td>varchar</td><td>Учитель начальных классов</td></tr>
# <tr><td>qualification</td><td>Квалификация</td><td>varchar</td><td>Экономист</td></tr>
# <tr><td>otherCertificates</td><td>Иные сертификаты соискателя</td><td>varchar</td><td>Справка об отсутствии судимости, сертификат COVID</td></tr>
# <tr><td>additionalEducationList</td><td>Информация о прохождении курсов</td><td> - </td><td> - </td></tr>
# <tr><td>additionalEducation</td><td>Дополнительное образование</td><td> - </td><td> - </td></tr>
# <tr><td>graduateYear</td><td>Год окончания</td><td>data</td><td>2017-12-01</td></tr>
# <tr><td>type</td><td>Тип</td><td>varchar</td><td>Course</td></tr>
# <tr><td>courseName</td><td>Название курса</td><td>varchar</td><td>Курсы для педагогов</td></tr>
# <tr><td>hardSkills</td><td>Профессиональные навыки</td><td> - </td><td> - </td></tr>
# <tr><td>hardSkill</td><td>Профессиональные навыки</td><td> - </td><td> - </td></tr>
# <tr><td>softskills</td><td>Гибкие навыки</td><td> - </td><td> - </td></tr>
# <tr><td>softskill</td><td>Гибкие навыки</td><td> - </td><td> - </td></tr>
# <tr><td>codeExternalSystem</td><td>Код внешней системы</td><td>varchar</td><td>CZN</td></tr>
# <tr><td>country</td><td>Страна проживания</td><td> - </td><td> - </td></tr>
# <tr><td>countryName</td><td>Название станы проживания</td><td>varchar</td><td>Российская Федерация</td></tr>
# <tr><td>countryCode</td><td>Код страны</td><td>varchar</td><td>255</td></tr>
# <tr><td>scheduleType</td><td>Желаемый график работы</td><td>varchar</td><td>Полный рабочий день</td></tr>
# <tr><td>salary</td><td>Желаемая зарплата</td><td>int</td><td>22000</td></tr>
# <tr><td>desirableRelocationRegions</td><td>Желаемые регионы переезда</td><td> - </td><td> - </td></tr>
# <tr><td>RelocationRegions</td><td>Регион переезда</td><td> - </td><td> - </td></tr>
# <tr><td>regionCode</td><td>Код региона</td><td>varchar</td><td>2400000000000</td></tr>
# <tr><td>type</td><td>Тип</td><td>varchar</td><td>CandidateRelocation</td></tr>
# <tr><td>busyType</td><td>Тип занятости</td><td>varchar</td><td>Полная занятость</td></tr>
# <tr><td>relocation</td><td>Готовность к переезду</td><td>varchar</td><td>Нет</td></tr>
# <tr><td>languageKnowledge</td><td>Знание иностранных языков</td><td> - </td><td> - </td></tr>
# <tr><td>languageKnowledge</td><td>Знание иностранных языков</td><td> - </td><td> - </td></tr>
# <tr><td>codeLanguage</td><td>Язык</td><td>varchar</td><td>Английский</td></tr>
# <tr><td>level</td><td>Уровень владения</td><td>varchar</td><td>BASIC</td></tr>
# <tr><td>isPreferred</td><td>Родной язык</td><td>boolean</td><td>false</td></tr>
# <tr><td>type</td><td>Тип</td><td>varchar</td><td>LanguageKnowledge</td></tr>
# <tr><td>businessTrips</td><td>Готовность к командировкам</td><td>varchar</td><td>Не готов к командировкам</td></tr>
# <tr><td>retrainingCapability</td><td>Возможность переобучения</td><td>varchar</td><td>Готов к переобучению</td></tr>
# <tr><td>innerinfo</td><td>Дополнительная информация</td><td> - </td><td> - </td></tr>
# <tr><td>idUser</td><td>Код пользователя</td><td>timeuuid</td><td>a296d280-352a-11e5-abfd-1ff705945672</td></tr>
# <tr><td>rfCitizen</td><td>Признак гражданства РФ</td><td>boolean</td><td>true</td></tr>
# <tr><td>visibility</td><td>Видимость</td><td></td><td>Видно всем</td></tr>
# <tr><td>dateModify</td><td>Дата изменения</td><td>date</td><td>2020-03-26</td></tr>
# <tr><td>deleted</td><td>Признак удаления</td><td>boolean</td><td>false</td></tr>
# <tr><td>fullnessRate</td><td>Процент заполнения</td><td>int</td><td>76</td></tr>
#
# </table>

# %% [markdown] id="amSq6T7aoj9k"
# Удалим все резюме, в которых не указана ожидаемая должность и размер зарплаты.

# %% id="pzoO6O4XpZPQ"
df = df[df['salary'].notna()]
df = df[df['positionName'].notna()]

# %% colab={"base_uri": "https://localhost:8080/"} id="nREjw7wjoy2A" outputId="9f160d2f-cf86-461c-b406-c8edee7a88b2"
df.info()


# %% [markdown] id="4ctVVdpX0OYd"
# Некоторые колонки имеют формат JSON (вложенная структура), сохраненные как текст. Чтобы с ними работать нам необходимо их преобразовать из текста в JSON:

# %% id="7EPspvxPxugR"
def load_json(js):
  try:
    return json.loads(js)
  except:
    return  []


# %% id="wNwj6COpUsYS"
df['workExperienceList']      = df['workExperienceList'].apply(load_json)
df['educationList']           = df['educationList'].apply(load_json)

# %% [markdown] id="pD-rehzygPw7"
# Мы видим, что в таблице 40 столбцов. В качестве целевого параметра, который будет предсказывать нейронная сеть, выберем колонку со значением желаемой зарплаты `salary`.
#
# Теперь нам необходимо из датасета извлечь данные, которые мы считаем значимыми для обучения модели, и представить их в формате пригодном для обучения нашей модели.

# %% [markdown] id="D4OCu9QjAkYN"
# ### Извлечение классов из данных

# %% [markdown] id="l46VnZUVAuGB"
# #### Извлечение города

# %% [markdown] id="dEJifftjBXit"
# Нам необходимо научиться извлекать из данных город проживания нашего соискателя, так как в разных регионах страны будут разные уровни зарплат и, соответственно, разные ожидания у соискателей. Города разделим на 4 класса: Москва, Санкт-Петербург, города-миллионники и остальные города. А также учтем варианты написания городов, а точнее областей, в датасете.  

# %% id="wEepw_7sAugE"
city_class =  {'Московская-область'          : 0,
               'г-Москва'                    : 0,
               'Ленинградская-область'       : 1,
               'г-Санкт-Петербург'           : 1,
               'Новосибирская-область'       : 2,
               'Свердловская-область'        : 2,
               'Татарстан-республика'        : 2,
               'Нижегородская-область'       : 2,
               'Красноярский-край'           : 2,
               'Челябинская-область'         : 2,
               'Самарская-область'           : 2,
               'Башкортостан-республика'     : 2,
               'Ростовская-область'          : 2,
               'Краснодарский-край'          : 2,
               'Омская-область'              : 2,
               'Воронежская-область'         : 2,
               'Пермский-край'               : 2,
               'Волгоградская-область'       : 2,
               'Прочие-города'               : 3
              }


# %% [markdown] id="EHbokpXNEpWX"
# Ранее мы обсуждали, что НС хорошо работают с данными представленными в формате `one hot encoding` (`OHE`), поэтому распределим данные по категориям городов соискателей и получим вектора распределения `OHE`.

# %% id="FOpzG-AgDB0Q"
#  Преобразование информации о городе в one hot encoding

def city2OHE(param):
    # Определение размерности выходного вектора, как число уникальных классов
    num_classes = len(set(city_class.values()))

    # Если не смогли распарсить, то поле не заполнено
    # Устанавливаем значение по умолчанию (последний элемент в словаре)
    if not isinstance(param, str):
        param = list(city_class.keys())[-1]

    # Разбиваем строку на слова
    split_array = re.split(r'[ ,.:()?!]', param)

    # Поиск города в строке и присвоение ему класса
    for word in split_array:
        city_cls = city_class.get(word, -1)
        if city_cls >= 0:
            break
    else:
        # Обратите внимание, что у for имеется интересная конструкция for/else
        # Город не в city_class - значит его класс "Прочие-города"
        city_cls = num_classes - 1

    # Возврат в виде one hot encoding-вектора
    return utils.to_categorical(city_cls, num_classes)


# %% [markdown] id="8LU_Pg79Fbv5"
# Обратите внимание, что мы используем конструкцию `for/else`. Блок `else` в конструкции `for/else` будет выполнен, если цикл полностью завершил итерацию, но не будет выполнен, если цикл прерван оператором `break`. Эта конструкция позволит все не найденные города отнести к 3 классу `прочие города`.

# %% [markdown] id="NNSpjBkZGVTp"
# Приведем произвольный пример преобразования из датасета по полю `localityName` (*Наименование населенного пункта*):

# %% colab={"base_uri": "https://localhost:8080/"} id="V7RSLOizDC-M" outputId="c16642cc-056a-41b8-8b98-3c4be917bf84"
N = 8
print('Наименование населенного пункта: ', df.localityName[N])
print('Наименование населенного пункта в формате OHE:', city2OHE(df.localityName[N]))

# %% [markdown] id="msaUSdIFLv4C"
# ### Извлечение возраста и стажа

# %% [markdown] id="YnUIFmet2ftV"
# Возраст и стаж соискателя мы также превратим в one hot encoding (OHE) формат. Для этого зададим пороговые значения:

# %% id="RZZnC6wW2S60"
# Список порогов возраста
age_class = [18, 25, 32, 39, 46, 53, 60]

# Список порогов опыта работы лет
experience_class = [1, 3, 5, 7, 10, 15]


# %% [markdown] id="viqDTJW4534Q"
# Определим универсальную функцию перевода числа в диапазон OHE:

# %% id="V6vaQD936Ne3"
def range2OHE(param, class_list):

   # Определение размерности выходного вектора, как число уникальных классов
    num_classes = len(class_list)+1

    # Поиск интервала для входного значения
    for i in range(num_classes - 1):
        if float(param) < class_list[i]:
            cls = i                       # Интервал найден, выбор класса
            break
    else:
        cls = num_classes - 1             # Интервал не найден, выбор последнего класса

    # Возврат в виде one hot encoding-вектора
    return utils.to_categorical(cls, num_classes)


# %% colab={"base_uri": "https://localhost:8080/"} id="VcG3wuig82SG" outputId="07fd749b-0ec5-4a14-f6c5-7e78264da415"
N = 8
print('Стаж работы: ', df.experience[N])
print('Стаж работы в формате OHE: ',range2OHE(df.experience[N], experience_class))

print('Возраст соискателя: ', df.age[N])
print('Возраст соискателя в формате OHE: ', range2OHE(df.age[N], age_class))

# %% [markdown] id="LQarexL22Ohz"
# ### Извлечение графика работы и типа занятости

# %% [markdown] id="-RKLM6-q-a_i"
# Для того чтобы правильно сформировать классы, нам необходимо вывести все возможные уникальные значения требуемого столбца нашего DataFrame.

# %% [markdown] id="MxSznbibAL2N"
# Для типа занятости, где возможно только одно значение список довольно лаконичный:

# %% colab={"base_uri": "https://localhost:8080/"} id="gO-zF9H8_6Hj" outputId="cb6ae9f6-3aaa-4004-9610-6c442aeb623d"
df.busyType.apply(lambda x: x if isinstance(x, str) else 'Полная-занятость').unique()

# %% [markdown] id="n1PZQAoQTSHZ"
# Если занятость не задана, т.е. тип значения отличен от строки (`isinstance(x, str)`), тогда для таких значений проставляем тип `Полная-занятость`, считаем его значением по умолчанию.

# %% [markdown] id="zrfjrHvJCy85"
# В резюме соискателя может быть указано несколько различных типов графика работы:

# %% colab={"base_uri": "https://localhost:8080/"} id="ofaB_aGN_8Rm" outputId="c7ce25cf-533d-4a02-8a81-0f526abf2f8a"
df.scheduleType.apply(lambda x: x if isinstance(x, str) else 'Полный-рабочий-день').unique()

# %% [markdown] id="2bS7oV_OC_Mw"
# Извлечем уникальные значения:

# %% colab={"base_uri": "https://localhost:8080/"} id="D9nC8h0_IzJR" outputId="33569bb1-a6de-4338-f782-a3620b4c36f0"
unique_list = df.scheduleType.apply(lambda x: x if isinstance(x, str) else 'Полный-рабочий-день').unique()

shedule_list = []

for uniq_str in unique_list:
  row_array = uniq_str.replace(' ', '').split(',')
  shedule_list = list(set(shedule_list + row_array))

print(shedule_list)

# %% [markdown] id="yd9wxr1_Ul85"
# Здесь как и для типа занятости, все не строковые значения заменяем на 'Полный-рабочий-день'.

# %% [markdown] id="OFPKjHhvU27m"
# В итоге для типов занятости и графиков работы мы получаем следующие возможные значения (классы):

# %% id="TpXdbhGV-bcM"
# Типы занятости
employment_class = {
                    'Стажировка'          : 0,
                    'Временная'           : 1,
                    'Сезонная'            : 2,
                    'Частичная-занятость' : 3,
                    'Удаленная'           : 4,
                    'Полная-занятость'    : 5
                   }

# Графики работы
schedule_class = {
                  'Сменный-график'              : 0,
                  'Ненормированный-рабочий-день': 1,
                  'Вахтовый-метод'              : 2,
                  'Гибкий-график'               : 3,
                  'Неполный-рабочий-день'       : 4,
                  'Полный-рабочий-день'         : 5,

                 }


# %% id="3JfDwaXuWU_a"
# Общая функция преобразования строки к multi-вектору
# На входе данные и словарь сопоставления подстрок классам

def str2multiOHE(param, class_dict):

    # Определение размерности выходного вектора, как число уникальных классов
    num_classes = len(set(class_dict.values()))

    # Создание нулевого вектора
    result = np.zeros(num_classes)

    # Если не смогли распарсить, то поле не заполнено
    # Устанавливаем значение по умолчанию (последний элемент в словаре)
    if not isinstance(param, str):
        param = list(class_dict.keys())[-1]

    # Поиск значения в словаре и, если нашли, то проставляем 1 в найденной позиции
    for value, cls in class_dict.items():
        if value in param:
            result[cls] = 1.

    return result


# %% [markdown] id="3I5ZF4_qWTM3"
# Сформируем OHE для произвольного резюме:

# %% colab={"base_uri": "https://localhost:8080/"} id="kq1MXe87Xi4t" outputId="fb60ec70-2410-4a2d-a7b8-2c7e07918633"
N = 154
print('Тип занятости: ', df.busyType[N])
print('Тип занятости в формате OHE: ', str2multiOHE(df.busyType[N], employment_class))
print()
print('График работы: ', df.scheduleType[N])
print('График работы в формате OHE: ', str2multiOHE(df.scheduleType[N], schedule_class))

# %% [markdown] id="-iM6R6W6dv8A"
# Если указано несколько значений, то в каждом классе, к которому подходят значения, проставляем `1`. `nan` в данном датасете означает незаполненное поле. Для таких полей задаем значение по умолчанию (последний класс в словаре).

# %% [markdown] id="z6t_PJXAm2Mv"
# ### Обучающая выборка по числовым данным

# %% [markdown] id="hDqNqG9Wm-2r"
# Прежде чем мы приступим к обработке текстовых данных, давайте сформируем обучающую выборку по уже извлеченным числовым данным в формате OHE.

# %% id="MsQeih6TnZ5A"
# Фиксация индексов столбцов
COL_LOCALITY     = df.columns.get_loc('localityName')
COL_EXPERIENCE   = df.columns.get_loc('experience')
COL_AGE          = df.columns.get_loc('age')
COL_BUSY         = df.columns.get_loc('busyType')
COL_SCHED        = df.columns.get_loc('scheduleType')
COL_SALARY       = df.columns.get_loc('salary')


def get_row_data(row):
    # Объединение всех входных данных в один общий вектор
    x_data = np.hstack([
                city2OHE(row[COL_LOCALITY]),
                range2OHE(row[COL_EXPERIENCE], experience_class),
                range2OHE(row[COL_AGE], age_class),
                str2multiOHE(row[COL_BUSY], employment_class),
                str2multiOHE(row[COL_SCHED], schedule_class)
              ])

    # Вектор зарплат в тысячах рублей
    y_data = np.array([row[COL_SALARY]]) / 1000


    return x_data, y_data

def get_train_data(dataFrame):
    x_data = []
    y_data = []

    for row in dataFrame.values:
        x, y = get_row_data(row)
        x_data.append(x)
        y_data.append(y)

    return np.array(x_data), np.array(y_data)



# %% id="qnwaMvr3nWEB"
# Формирование выборки из загруженного набора данных
x_train, y_train = get_train_data(df)


# %% [markdown] id="slMvGQ7CeRr5"
# ### Извлечение данных об образовании

# %% [markdown] id="-wsmR3ZB0XgN"
# Сведения об образовании мы передадим в модель в виде текста, поэтому создадим новую колонку `education` в нашем наборе данных и поместим в нее сведения об учебном заведении, год окончания, факультет, специальность и полученная квалификация:

# %% id="kOgHwLnm3sh1"
def extract_education(param):
  edu_text = []
  for edu in param:
    if edu.get('instituteName'):
        edu_text.append(edu.get('instituteName'))
        if  edu.get('qualification'):
            edu_text.append(edu.get('qualification'))
        if  edu.get('specialty'):
            edu_text.append(edu.get('specialty'))
        if  edu.get('faculty'):
            edu_text.append(edu.get('faculty'))
        if  edu.get('graduateYear'):
            edu_text.append(str(edu.get('graduateYear')))


  return '. '.join(edu_text)


df['education'] = df['educationList'].apply(extract_education)

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="24CPCBzX5j80" outputId="b4d34a29-e050-4d24-a579-317b373b3f42"
df['education'][5]

# %% [markdown] id="K4lD3cHX__J4"
# Преобразуем текстовые данные в числовые для обучения нейросети:
#

# %% id="6d4m9_Og_Un3"

# Используется встроенный в Keras токенизатор для разбиения текста и построения частотного словаря
tokenizer = Tokenizer(
    num_words=3000,                                          # объем словаря
    filters='!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', # убираемые из текста ненужные символы
    lower=True,                                              # приведение слов к нижнему регистру
    split=' ',                                               # разделитель слов
    oov_token='unknown',                                     # токен для слов, которые не вошли в словарь
    char_level=False                                         # разделяем по словам, а не по единичным символам
)

# Построение частотного словаря по текстам образования
tokenizer.fit_on_texts(df['education'])

# %% colab={"base_uri": "https://localhost:8080/"} id="89zH6MzB_VeS" outputId="3df0c88b-d5a8-45ca-85e8-6d9b6d2b99d2"
items = list(tokenizer.word_index.items())       # Получение индексов слов
print(items[:50])                                # Посмотр 50 самых часто встречающихся слов
print("Размер словаря", len(items))              # Длина словаря

# %% id="32l_ER-S_YP8"
# Преобразование текстов в последовательность индексов согласно частотному словарю
education_seq = tokenizer.texts_to_sequences(df['education'])

# %% id="87sJTeSk_f7K"
# Преобразование последовательностей индексов в bag of words
x_train_education = tokenizer.sequences_to_matrix(education_seq)

# %% colab={"base_uri": "https://localhost:8080/"} id="7sRhz7hW_ghs" outputId="8f93cf46-7ec6-416d-9fc3-04352660fab5"
# Проверка результата
print(x_train_education.shape)
print(x_train_education[5][0:100])

# %% colab={"base_uri": "https://localhost:8080/"} id="agtgG7X3_lBv" outputId="94ff7e69-bdc2-4966-8a77-0293ede9f953"
# Проверка получившихся данных
n = 5
print(df['education'][n])                   # Данные об образовании в тексте
print(education_seq[n])                     # Данные об образовании в индексах слов
print(x_train_education[n][0:100])          # Данные об образовании в bag of words

# %% id="xo_TszFy_s-W"
# Освобождение памяти от промежуточных данных
del education_seq, tokenizer

# %% [markdown] id="fgugo7tz_tvR"
# ### Извлечение данных о предыдущей работе

# %% [markdown] id="Ei_PCTr5J61R"
# Извлечение данных о предыдущих местах работы соискателя производится по аналогичной схеме, что мы делали выше при извлечении образовательных учреждений соискателя.

# %% colab={"base_uri": "https://localhost:8080/", "height": 140} id="pbv2w5kQJ7iV" outputId="8ba0c07d-6813-44cc-8af5-03c1db854a8b"
# Библиотека для работы с датой и временем
# https://docs.python.org/3/library/datetime.html - ссылка на документацию
import datetime

def extract_works(param):
  edu_text = []
  for edu in param:
    if edu.get('companyName'):
        edu_text.append(edu.get('companyName'))
        if  edu.get('dateFrom') and edu.get('dateTo'):
            # Преобразуем строку в формат даты и времени (дата увольнения)
            dateT = datetime.datetime.strptime(edu.get('dateTo'), '%Y-%m-%dT%H:%M:%S%z')

            # Преобразуем строку в формат даты и времени (дата приема на работу)
            dateF = datetime.datetime.strptime(edu.get('dateFrom'), '%Y-%m-%dT%H:%M:%S%z')

            # разницу дат делим на число секунд в месяц
            edu_text.append(f"Стаж {int((dateT - dateF).total_seconds() / 2628000) } месяцев")

        if  edu.get('jobTitle'):
            edu_text.append(edu.get('jobTitle'))
        if  edu.get('achievements'):
            edu_text.append(edu.get('achievements'))
        if  edu.get('demands'):
            edu_text.append(edu.get('demands').replace('<p>', '').replace('</p>', ''))



  return '. '.join(edu_text)


df['works'] = df['workExperienceList'].apply(extract_works)

df['works'][9]

# %% id="f7Z91646TODw"
# Используется встроенный в Keras токенизатор для разбиения текста и построения частотного словаря
tokenizer = Tokenizer(
    num_words=3000,                                          # объем словаря
    filters='!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', # убираемые из текста ненужные символы
    lower=True,                                              # приведение слов к нижнему регистру
    split=' ',                                               # разделитель слов
    oov_token='unknown',                                     # токен для слов, которые не вошли в словарь
    char_level=False                                         # разделяем по словам, а не по единичным символам
)

# Построение частотного словаря по текстам с опытом работы
tokenizer.fit_on_texts(df['works'])

# %% id="6Gs_hZa8Y6RX" colab={"base_uri": "https://localhost:8080/"} outputId="1eb5820f-59b7-42ce-d0c2-8f85e7f2e211"
items = list(tokenizer.word_index.items())       # Получение индексов слов
print(items[:50])                                # Посмотр 50 самых часто встречающихся слов
print("Размер словаря", len(items))              # Длина словаря

# %% id="6seOHdiuY-pY" colab={"base_uri": "https://localhost:8080/"} outputId="2b0b1323-c6a0-4c49-f992-ad1dd6f6bb59"
# Преобразование текстов в последовательность индексов согласно частотному словарю
works_seq = tokenizer.texts_to_sequences(df['works'])

# Преобразование последовательностей индексов в bag of words
x_train_works = tokenizer.sequences_to_matrix(works_seq)

print('Форма обучающей выборке по опыту работы:', x_train_works.shape)
print()

# Проверка получившихся данных
n = 5
print(df['works'][n])                      # Опыт работы в тексте
print(works_seq[n])                        # Опыт работы в индексах слов
print(x_train_works[n][0:100])             # Опыт работы в bag of words

# %% id="FQv6XnVgY_JR"
# Освобождение памяти от промежуточных данных
del works_seq, tokenizer

# %% [markdown] id="mghCS7-xc49j"
# ### Извлечение данных о желаемой Должности

# %% [markdown] id="otFcmo4xc49j"
# Данные о должности хранятся в столбце `positionName` и не требуют изменений и сложных извлечений. Однако для передачи данных в нейронную сеть их тоже необходимо привести к формату `bag of words`.

# %% id="GKZiSJRqR1Jh"

pd.Series(df['positionName'].unique()).to_csv('test.csv')

# %% id="A0bHRAY5c49j"
# Используется встроенный в Keras токенизатор для разбиения текста и построения частотного словаря
tokenizer = Tokenizer(
    num_words=3000,                                          # объем словаря
    filters='!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\r\xa0', # убираемые из текста ненужные символы
    lower=True,                                              # приведение слов к нижнему регистру
    split=' ',                                               # разделитель слов
    oov_token='unknown',                                     # токен для слов, которые не вошли в словарь
    char_level=False                                         # разделяем по словам, а не по единичным символам
)

# Построение частотного словаря по текстам с опытом работы
# Мы используем принудительное преобразование данных к строке, чтобы избежать ошибок в случае пропуска данных
tokenizer.fit_on_texts(df['positionName'].apply(str))

# %% colab={"base_uri": "https://localhost:8080/"} outputId="794da3c1-50a1-4f9d-f664-bc70a700ae5d" id="o6QJZBtwc49k"
items = list(tokenizer.word_index.items())       # Получение индексов слов
print(items[:50])                                # Посмотр 50 самых часто встречающихся слов
print("Размер словаря", len(items))              # Длина словаря

# %% colab={"base_uri": "https://localhost:8080/"} outputId="bba90458-f088-4e3f-e04a-88d0ebecbcb5" id="1v1elf4Xc49k"
# Преобразование текстов в последовательность индексов согласно частотному словарю
position_seq = tokenizer.texts_to_sequences(df['positionName'].apply(str))

# Преобразование последовательностей индексов в bag of words
x_train_position = tokenizer.sequences_to_matrix(position_seq)

print('Форма обучающей выборке по опыту работу:', x_train_position.shape)
print()

# Проверка получившихся данных
n = 5
print(df['positionName'][n])                  # Опыт работы в тексте
print(position_seq[n])                        # Опыт работы в индексах слов
print(x_train_position[n][0:100])             # Опыт работы в bag of words

# %% id="lBWR143ac49k"
# Освобождение памяти от промежуточных данных
del position_seq, tokenizer, items, df

# %% [markdown] id="7ay-ckqIEXBm"
# ### Нормализация зарплат

# %% [markdown] id="hbSeLDVHExm1"
# Мы уже говорили, что выходные данные, на которых обучается наша модель необходимо нормализовать, поэтому нормализуем уровень зарплат с помощью инструмента `StandardScaler`:

# %% id="PLfWWuNysekZ"
# Для нормализации данных используется готовый инструмент
y_scaler = StandardScaler()

# Нормализация выходных данных по стандартному нормальному распределению
y_train_scaled = y_scaler.fit_transform(y_train)

# %% [markdown] id="4Rpvqo6rc49k"
# ## Архитектура модели

# %% [markdown] id="N_gMHiLHj104"
# Давайте остановимся и немного поразмышляем какой должна быть модель, чтобы решить задачу регрессии с 4 наборами данных! Именно так у нас получилось 4 набора данных:
# * числовой набор - это совокупность наших OHE-векторов, которые мы подадим одним вектором на вход модели;
# * текстовый набор с данными об образовании;
# * текстовый набор с данными об опыте работы;
# * текстовый набор с данными о желаемой должности.
#
# Получается, что наша модель должна иметь 4 входа и один выходной регрессирующий нейрон, который может выдавать любое положительное значение на выходе из сети.
#

# %% id="xfNi9SSTlVta"
input1 = Input((x_train.shape[1],))
input2 = Input((x_train_education.shape[1],))
input3 = Input((x_train_works.shape[1],))
input4 = Input((x_train_position.shape[1],))

# Первый вход для числовых данных
x1 = input1
x1 = Dense(20, activation="relu")(x1)
x1 = Dense(500, activation="relu")(x1)
x1 = Dense(200, activation="relu")(x1)


# Второй вход для данных об образовании
x2 = input2
x2 = Dense(20, activation="relu")(x2)
x2 = Dense(200, activation="relu")(x2)
x2 = Dropout(0.3)(x2)

# Третий вход для данных об опыте
x3 = input3
x3 = Dense(20, activation="relu")(x3)
x3 = Dense(200, activation="relu")(x3)
x3 = Dropout(0.3)(x3)

# Четвертый вход для данных о желаемой должности
x4 = input4
x4 = Dense(20, activation="relu")(x4)
x4 = Dense(200, activation="relu")(x4)
x4 = Dropout(0.3)(x4)


# Объединение четырех веток
x = concatenate([x1, x2, x3, x4])

# Промежуточный слой
x = Dense(30, activation='relu')(x)
x = Dropout(0.5)(x)

# Финальный регрессирующий нейрон
x = Dense(1, activation='linear')(x)

# В Model передаются входы и выход
model = Model((input1, input2, input3, input4), x)

# %% [markdown] id="oubVZizqc49k"
# Построим схему нашей модели:

# %% colab={"base_uri": "https://localhost:8080/", "height": 768} id="A1144jaF7vNM" outputId="a28d73a6-bc8d-411d-9d46-67a9489dd359"
utils.plot_model(model, dpi=96, show_shapes=True, show_layer_activations=True)

# %% [markdown] id="UkTxQCPIGYvn"
# Мы видим, что модель состоит из четырех входов, где слои и гиперпараметры подобраны случайным образом и не оптимизированы, однако, логика в построении архитектуры все же соблюдалась.
#
# А именно:
# * добавленны слои регуляризации `Dropout`, на выходе из веток с текстовыми данными
# * выходы из каждой ветки перед операцией объединения веток (операция `concatenate` в Keras) имеют одинаковую размерность
# * выходной слой состоит из одного регрессирующего нейрона с линейной функцией активации, перед которым используется слой регуляризации `Dropout`.

# %% [markdown] id="SP_7_7PeWBO4"
# **Обучим модель**
#
# Обратите внимание, что `model.fit()` мы передаем в списке весь наш массив обучающих данных, в том же порядке, что мы определили в моделе `Model((input1, input2, input3, input4), x)`. В качестве функции потерь мы используем среднюю квадратичную ошибку (`mse`). В качестве метрики анализируем cреднюю абсолютную ошибку (`mae`), которая должна снижаться в процессе обучения.

# %% colab={"base_uri": "https://localhost:8080/"} id="w_oEW_Yd8djx" outputId="e8289ca8-5b85-4953-dcf3-613cf2f90fca"
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

history = model.fit([x_train[:8000], x_train_education[:8000], x_train_works[:8000], x_train_position[:8000]],
                           y_train_scaled[:8000],
                           batch_size=256,
                           epochs=100,
                           validation_data=([x_train[8000:], x_train_education[8000:], x_train_works[8000:], x_train_position[8000:]], y_train_scaled[8000:]),
                           verbose=1)

# %% [markdown] id="dQxM46PD0Qbm"
# **Визуализируем процесс обучения**

# %% [markdown] id="Gom2iZS81gX0"
# Средняя абсолютная ошибка показывает нам на сколько в абсолютных величинах ошибается наша модель. Если бы мы не нормализовали данные, то каждая точка на графике соответствовала бы среднему отклонению в тысячах рублей предсказанного значения от реального.

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} id="dBtC94118xw_" outputId="1aef8168-fc6c-4566-c050-c1bbb69342c7"
plt.plot(history.history['mae'], label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'], label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

# %% [markdown] id="V-gFxz4FfJ52"
# По графику обучения, мы видим, что ошибка продолжает снижаться, переобучения не наблюдается, можно и дальше продолжать обучение.

# %% [markdown] id="zGkwMDtub8Rj"
# Выведем 100 точек из проверочной выборки и отобразим их на диагональной линии ожидаемых значений. Также выведем среднюю абсолютную ошибку.

# %% colab={"base_uri": "https://localhost:8080/", "height": 773} id="DOiNFCS6_MlS" outputId="3cc00afd-3fc9-4d5e-957f-32ebdadd8fca"

pred = model.predict([x_train[8000:8100], x_train_education[8000:8100], x_train_works[8000:8100], x_train_position[8000:8100]])

pred = y_scaler.inverse_transform(pred)    # Обратная нормированию процедура

print('Средняя абсолютная ошибка:', mean_absolute_error(pred, y_train[8000:8100]), '\n') # расчет средней абсолютной ошибки

for i in range(10):
    print('Реальное значение: {:6.2f}  Предсказанное значение: {:6.2f}  Разница: {:6.2f}'.format(y_train[8000:8100][i, 0],
                                                                                                pred[i, 0],
                                                                                                abs(y_train[8000:8100][i, 0] - pred[i, 0])))
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_train[8000:8100], pred)          # Отрисовка точечного графика
ax.set_xlim(0, 100)                           # Ограничение оси по x
ax.set_ylim(0, 100)                           # Ограничение оси по x
ax.plot(plt.xlim(), plt.ylim(), 'r')          # Отрисовка диагональной линии
plt.xlabel('Правильные значения')
plt.ylabel('Предсказания')
plt.grid()
plt.show()

# %% [markdown] id="bF-Ub-0Zc49k"
# Значение средней абсолютной ошибки в 10.7, говорит о том, что наша модель в среднем ошибается на 10700 рублей при предсказании ожидаемой зарплаты от указанной в резюме.

# %% id="G5UJ5EqspFGH"
# Освобождение памяти от промежуточных данных
del history, model, pred

# %% [markdown] id="9Yk8eAhd6h7m"
# ## Упрощенная модель

# %% [markdown] id="rE0JEsDG60Rm"
# Рассмотрим еще одну упрощенную модель. Для упрощения модели сделаем следующее допущение: вся информация об опыте работы соискателя содержится в его общем опыте работы `experience` и должности, на которую претендует соискатель. Т.е., выставляя свое резюме, соискатель будет ориентироваться на цену по рынку, а не на свои прежние места работы и знания. Надо понимать, что резюме - это всего лишь пожелания соискателя, а не его реальная зарплата!
#
# При таком допущении нам достаточно использовать 2 ветки НС вместо 4-х. Первая ветка - это все наши числовые данные, а вторая ветка - текстовое значение должности.

# %% [markdown] id="ztSnaX6y9JqK"
# **Архитектура упрощенной модели**

# %% id="8mwjmizzVwh-"
input1 = Input((x_train.shape[1],))
input2 = Input((x_train_position.shape[1],))

# Первый вход для числовых данных
x1 = input1
x1 = Dense(20, activation="relu")(x1)
x1 = Dense(500, activation="relu")(x1)
x1 = Dense(200, activation="relu")(x1)

# Второй вход для данных о желаемой должности
x2 = input2
x2 = Dense(20, activation="relu")(x2)
x2 = Dense(200, activation="relu")(x2)
x2 = Dropout(0.3)(x2)


# Объединение четырех веток
x = concatenate([x1, x2])

# Промежуточный слой
x = Dense(30, activation='relu')(x)
x = Dropout(0.5)(x)

# Финальный регрессирующий нейрон
x = Dense(1, activation='linear')(x)

# В Model передаются входы и выход
model = Model((input1, input2), x)

# %% colab={"base_uri": "https://localhost:8080/", "height": 865} id="gy21mqs1WfyB" outputId="8e4b5d9d-b0da-42b9-b79e-173b1eb8c2c6"
utils.plot_model(model, dpi=96, show_shapes=True, show_layer_activations=True)

# %% [markdown] id="XnkVrKAh9Q3E"
# **Обучим модель**

# %% colab={"base_uri": "https://localhost:8080/"} id="UEcg-N0qWkhM" outputId="cc6cc35a-5d96-4817-c77e-ec8b19549aab"
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

history = model.fit([x_train[:8000], x_train_position[:8000]],
                           y_train_scaled[:8000],
                           batch_size=256,
                           epochs=100,
                           validation_data=([x_train[8000:], x_train_position[8000:]], y_train_scaled[8000:]),
                           verbose=1)

# %% [markdown] id="i1mWEDv-9Skn"
# **Визуализируем результат обучения**

# %% id="eHULwjQrW0aj" colab={"base_uri": "https://localhost:8080/", "height": 449} outputId="57a5b3ad-697d-45c0-dc0d-c24f05387569"
plt.plot(history.history['mae'], label='Средняя абсолютная ошибка на обучающем наборе')
plt.plot(history.history['val_mae'], label='Средняя абсолютная ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

# %% id="wsS2lmS2W_a5" colab={"base_uri": "https://localhost:8080/", "height": 773} outputId="936974b2-5ea7-49c8-c86b-b938fb36662e"
pred = model.predict([x_train[8000:8100], x_train_position[8000:8100]])  # Предсказание на новых данных (контрольный образец)

pred = y_scaler.inverse_transform(pred)    # Обратная нормированию процедура


print('Средняя абсолютная ошибка:', mean_absolute_error(pred, y_train[8000:8100]), '\n') # расчет средней абсолютной ошибки

for i in range(10):
    print('Реальное значение: {:6.2f}  Предсказанное значение: {:6.2f}  Разница: {:6.2f}'.format(y_train[8000:8100][i, 0],
                                                                                                pred[i, 0],
                                                                                                abs(y_train[8000:8100][i, 0] - pred[i, 0])))
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(y_train[8000:8100], pred)          # Отрисовка точечного графика
ax.set_xlim(0, 100)                           # Ограничение оси по x
ax.set_ylim(0, 100)                           # Ограничение оси по x
ax.plot(plt.xlim(), plt.ylim(), 'r')          # Отрисовка диагональной линии
plt.xlabel('Правильные значения')
plt.ylabel('Предсказания')
plt.grid()
plt.show()

# %% [markdown] id="nC0qz0r4glN8"
# Значение средней абсолютной ошибки в 10.4 для упрощенной модели, говорит о том, что наша модель в среднем ошибается на 10400, что немного лучше прежнего результата. Может показаться, что упрощенная модель лучше? Но это не совсем так. Более тяжелая модель с 4 ветками будет дольше обучаться. Тем более мы видим, что нет переобучения, что даже на 100 эпохах обе модели продолжают обучаться и дальше.  

# %% [markdown] id="Jq2QG-4YglDe"
# На этом наше знакомство с регрессионными моделями подошло к концу и пора приступить к выполнению [домашней работы](https://colab.research.google.com/drive/1iPTkGZ_AEUpl5l6DR__J021gHR61RRfQ).
