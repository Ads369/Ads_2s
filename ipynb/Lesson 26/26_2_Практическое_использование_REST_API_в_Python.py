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
# <a href="https://colab.research.google.com/github/Ads369/Ads_2s/blob/main/ipynb/Lesson%2026/26_2_%D0%9F%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D0%B8%D1%81%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_REST_API_%D0%B2_Python.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="n_TP2nih3N41"
# **Навигация по уроку**
#
# 1. [Веб-архитектура сервиса](https://colab.research.google.com/drive/10wtDodlf4SaVcYk6VoXDWk650IDcNPaa)
# 2. Практическое использование REST API в Python
# 3. [Введение в FastAPI](https://colab.research.google.com/drive/1_AzAVys4xub3yyw763NDwfeJ3WecGgkb)
# 5. [Домашняя работа](https://colab.research.google.com/drive/1SlJW51-OaUUDPuk-j9GYNHHwFkHBztU_)

# %% [markdown] id="k5h0imbILFsl"
# ## Реализация REST API в Python
#
# Наиболее популярными библиотеками для работы с **HTTP** в **Python** являются **urllib**, **urllib2**, **httplib** и **requests**. Для доступа к низкоуровневым функциям протокола используют **socket**.
#
# Самой удобной библиотекой принято считать **requests**, потому ее все чаще включают в свой инструментарий не только программисты, но и специалисты в области DataScience.
#
#
#
#

# %% [markdown] id="GB5ocFhjNHwI"
# ### Библиотека **REQUESTS**

# %% [markdown] id="G1InQ0EWMg6I"
# Рассмотрим реализацию основных методов **REST API** с использованием модуля **requests**.

# %% [markdown] id="VBXi6VQfNYw6"
# Попробуем получить веб-страницу с помощью **GET**-запроса.

# %% [markdown] id="S4Jti7_qJhyA"
# **GET** является одним из самых популярных **HTTP**-методов. Метод **GET** указывает на то, что происходит попытка извлечь данные из определенного ресурса. Для того, чтобы выполнить запрос **GET**, используется метод `requests.get()`.
#
# Для проверки работы команды мы будет использовать [Root REST API](https://docs.github.com/ru/rest?apiVersion=2022-11-28#root-endpoint) на GitHub. Для указанного ниже URL вызывается метод `get()`:

# %% id="3uZ9gTBkJi24"
import requests
response = requests.get('https://api.github.com')

# %% [markdown] id="PrHZeXjtJhus"
# В переменную `response` мы поместили значение объекта **Response** и теперь можем рассмотреть более детально данные, которые были получены в результате запроса **GET**.
#
#

# %% [markdown] id="TCy1bjNOQ7La"
# **HTTP коды состояний**

# %% [markdown] id="yFr520vJW5yn"
# Мы уже познакомились с кодами состояний в первой части урока и знаем, что если запрос был успешно выполнен, то получим код состояния `200`. Проверим данное утверждение. Для этого необходимо обратиться к атрибуту `status_code`, полученного объекта:

# %% colab={"base_uri": "https://localhost:8080/"} id="an5bC7wwSz7Z" outputId="1fba47cf-2de8-46ba-cd5f-bcb9b8a4a270"
response.status_code

# %% [markdown] id="EmwNNt7CSwYw"
#
# ```
#
# `status_code` вернул значение `200`. Это значит, что запрос был выполнен успешно!
#
# Код состояния часто используется, чтобы задать разную логику программы в зависимости от результата запроса:

# %% colab={"base_uri": "https://localhost:8080/"} id="hAhB49mmTlVO" outputId="d472c275-17dd-44ad-87eb-101aa21a437e"
if response.status_code == 200:
    print('Данные получены!')
elif response.status_code == 404:
    print('Упс! Попробуйте снова!')

# %% [markdown] id="NtTQYT6VToCN"
# Библиотека `requests` предоставляет возможность использовать и более простую конструкцию. Если использовать полученный объект `Response` в условных конструкциях, то при получении кода состояния в промежутке от  `200` до `400`, будет выведено значение `True`. В противном случае отобразится значение `False`.
#
# Последний пример можно упростить при помощи использования оператора `if`:

# %% colab={"base_uri": "https://localhost:8080/"} id="sF3SAp37TydR" outputId="83ddeecf-89b2-414f-9073-426da5fb3564"
if response:
    print('Данные получены!')
else:
    print('Упс! Попробуйте снова!')

# %% [markdown] id="kZ0r5myWY2Af"
# Стоит иметь в виду, что данный способ не проверяет, имеет ли статусный код точное значение `200`. Причина заключается в том, что другие коды в промежутке от `200` до `400`, например, `204 NO CONTENT` и `304 NOT MODIFIED`, также считаются успешными в случае, если они могут предоставить действительный ответ.
#
# К примеру, код состояния `204` говорит о том, что ответ успешно получен, однако в полученном объекте нет содержимого. Можно сказать, что для оптимально эффективного использования способа необходимо убедиться, что начальный запрос был успешно выполнен. Требуется изучить код состояния и в случае необходимости произвести необходимые поправки, которые будут зависеть от значения полученного кода.
#
# Если вы не хотите использовать оператора `if` для проверки кода состояния, то можете генерировать исключения при неудачных запросах с помощью метода `raise_for_status()`:

# %% colab={"base_uri": "https://localhost:8080/"} id="qyiTKeToZVxf" outputId="dc4aa52c-5b13-4fac-8d33-fdc2980e370f"
import requests
# импорт ошибки метода HTTP
from requests.exceptions import HTTPError

# В цикле обращаемся к двум URI: реальному и вымышленному
for url in ['https://api.github.com', 'https://api.github.com/invalid']:
    print(f'Запрос по адресу: {url}')
    # Блок обработки исключений (ошибок)
    try:
        # выполняем GET запрос
        response = requests.get(url)

        # если ответ успешен, исключения задействованы не будут
        response.raise_for_status()
    except HTTPError as http_err:
        # Если ошибка связана с HTTP запросом, то выполнится этот блок HTTPError
        print(f'HTTP ошибка: {http_err}')
    except Exception as err:
        # Если ошибка не связана с HTTP запросом, то выполнится этот блок Exception
        print(f'Другая ошибка (не HTTP): {err}')
    else:
        # При успешном выполнении после блока try выполнится данный блок else
        print('Это успех!')

# %% [markdown] id="XIJe5CPw-BEu"
# Таким образом при вызова метода `raise_for_status()` проверяется код состояния, и для некоторых кодов вызывается исключение `HTTPError`. Если код состояния относится к успешным, то программа продолжает выполнение.

# %% [markdown] id="gfraQldx_Pg4"
# **Содержимое страницы**

# %% [markdown] id="57qRlrfl_Wyg"
# Делая GET запрос, мы ожидаем получить ценную для нас информацию. Эта информация, как правило, находится в теле сообщения и называется **пейлоад (payload)**. Используя атрибуты и методы объекта **Response**, можно извлечь информацию из пайлоад в различных форматах.

# %% [markdown] id="1nVFEY7x_aJI"
# Для того, чтобы получить содержимое запроса в байтах, необходимо обратиться к атрибуту `content`:

# %% colab={"base_uri": "https://localhost:8080/"} id="i3TDcunW_c_D" outputId="89c0c9af-c276-4244-9841-a3dbb9c04f58"
response.content

# %% [markdown] id="nCunxMyr_nGB"
# Использование `content` обеспечивает доступ к чистым байтам ответного пейлоада, то есть к любым данным в теле запроса. Однако, зачастую требуется получить информацию в виде строки в кодировке `UTF-8`, что можно сделать, обратившись к атрибуту `text`:

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="TrsEoTPE_g_b" outputId="b03d2dd5-862c-4625-e2e9-7170a90f10cf"
response.text

# %% [markdown] id="ryDTIhKA_o-C"
# Декодирование байтов в строку требует наличия определенной кодировки. По умолчанию **requests** попытается узнать текущую кодировку, ориентируясь по заголовкам HTTP. Указать необходимую кодировку можно при помощи добавления `encoding` перед `text`:

# %% id="SEXKDeM4_tKQ"
response.encoding = 'utf-8' # Сообщаем, что нам нужна кодировка utf-8
response.text

# %% [markdown] id="5eWIrdZH_yOQ"
# Если присмотреться к ответу, то можно заметить, что его содержимое является сериализированным (преобразованным в текст) JSON контентом. Воспользовавшись библиотекой `json`, можно взять полученные из `text` строки `str` и провести с ними обратную сериализацию (преобразовать в JSON) при помощи использования `json.loads()`:

# %% colab={"base_uri": "https://localhost:8080/"} id="7D7CgqF6Fj3F" outputId="1ec789f3-0e14-4827-90f6-1129642cef33"
import json
json.loads(response.text)

# %% [markdown] id="JJ5N4--uFedx"
# Есть и более простой способ с использованием  метода `json()`:

# %% colab={"base_uri": "https://localhost:8080/"} id="2R4JQXRUFe1W" outputId="b433b3f8-37e2-4090-a41c-2df702764879"
response.json()

# %% [markdown] id="FKKJLBDY_oq_"
# Тип полученного значения методом `json()`, является словарем. А значит, доступ к его содержимому можно получить по ключу.

# %% [markdown] id="9MW6I_RO_73t"
# **HTTP-заголовки**

# %% [markdown] id="yqwVx_JvCoQ5"
# **HTTP-заголовки** ответов на запрос могут содержать полезную информацию. Это может быть тип содержимого ответного пейлоада, либо ограничение по времени для кеширования ответа и многое другое.
#
# Для просмотра HTTP заголовков необходимо обратиться к атрибуту `headers`:
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="7PXpOImmKt4N" outputId="9d754fb8-9c27-4bc5-ecd1-7adaf57a141a"
response.headers

# %% [markdown] id="ILNnftZ1Covn"
# Ответ в формате "словаря" (почти), в таком виде сложно читается. Для упрощения восприятия можно сериализовать данные с помощью `json.dumps()` с параметром `indent` (число отступов в пробелах при форматировании):

# %% colab={"base_uri": "https://localhost:8080/"} id="OwJ9Lai9LN4k" outputId="d4cdc439-7942-449f-de40-c22e9113c8a7"
print(json.dumps(dict(response.headers), indent=4))

# %% [markdown] id="HJagrN1vMtz1"
# Прежде чем значение `response.headers` сериализовать, мы его привели к типу данных `dict`, так как он имеет тип отличный от привычного нам словаря:

# %% colab={"base_uri": "https://localhost:8080/", "height": 186} id="iCop9L7dMoZ9" outputId="6a14ba05-85d0-4507-ef12-da74053a8171"
type(response.headers)

# %% [markdown] id="Q44_bJ3PNbEi"
# Он похож на словарь, и многие методы к нему применимы, как со словарем, но когда доходит дело до сериализации (то есть преобразование к JSON подобному тексту), должен быть на входе именно словарь (а не словарь подобный объект).  

# %% [markdown] id="t5kWIUxATxK9"
# Так как `headers` возвращает объект подобный словарю, то мы можем получить доступ к значению заголовка HTTP по ключу. Например, для просмотра типа содержимого ответного пейлоада, требуется использовать `Content-Type`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="MUFkqB8wUQ59" outputId="a1cdc4ab-7013-4e85-fbce-7f069292e42a"
response.headers['Content-Type']

# %% [markdown] id="cdbBiEdHUP94"
# Специфика HTTP предполагает, что заголовки не чувствительны к регистру. Это значит, что при получении доступа к заголовкам можно не беспокоится о том, использованы строчные или прописные буквы.

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="JZUcebkrTwuP" outputId="1aaf6738-65c4-4708-b643-ba68ae78af20"
response.headers['content-type']

# %% [markdown] id="aJmLXE-nUfpo"
# При использовании ключей `content-type` и `Content-Type` результат будет получен один и тот же. Таким свойством обычный словарь не обладает!

# %% [markdown] id="-ZdQP9M_VVVp"
# **Параметры запроса**

# %% [markdown] id="8tAhMtDPWIhN"
# Наиболее простым способом выполнить запрос **GET** с параметрами является передача значений через параметры строки запроса в URI. При использовании метода `get()`, данные передаются в `params`:

# %% colab={"base_uri": "https://localhost:8080/"} id="_flhwiEBWqSK" outputId="a8aa4207-ffd4-4887-f802-2502dd57a803"
import requests

# Поиск репозитариев c упоминание ключевого слова requests и языка Python на GitHub
response = requests.get(
    'https://api.github.com/search/repositories', # URI
    params={'q': 'requests+language:python'},     # Параметры для запроса
)

# Данные в формате JSON
json_response = response.json()

# Первый найденный репозитарий
repository = json_response['items'][0]


print(f'Имя репозитария: {repository["name"]}')
print(f'Описание репозитария: {repository["description"]}')


# %% [markdown] id="OFg6D-f7bhxa"
# Параметры можно передавать в форме словаря:
# ```python
# params={'q': 'requests+language:python'}
# ```
#
# в форме списка кортежей:
# ```python
# params=[('q', 'requests+language:python')]
# ```
#
# или передать значение в байтах:
# ```python
# params=b'q=requests+language:python'
# ```

# %% [markdown] id="LC0fua0aWI-F"
# **Настройка HTTP-заголовка запроса (headers)**

# %% [markdown] id="z2oxwNJdWMnZ"
# Для изменения HTTP-заголовка требуется передать словарь данного HTTP-заголовка в `get()` при помощи использования параметра `headers`.
#
# Например, можно взять предыдущий пример, и указать ГитХабу (GitHub), что необходимо подсветить в ответе все места с найденным текстом как в запросе (`requests+language:python`).
#
# Для этого в заголовке `Accept` передается тип `text-match`, понятный ГитХабу.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="9N-63Rn-dNCv" outputId="1997be65-424c-411a-b1e6-0a04ba547c7b"
import requests

response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
    headers={'Accept': 'application/vnd.github.v3.text-match+json'},
)

# просмотр нового массива `text-matches` с предоставленными данными
# о поиске в пределах результатов
json_response = response.json()
repository = json_response['items'][0]
repository["text_matches"]

# %% [markdown] id="x-rvVqg1WMkA"
# Заголовок `Accept` сообщает серверу о типах контента, который можно использовать в рассматриваемом приложении. Здесь подразумевается, что все совпадения будут подсвечены, для чего в заголовке используется значение `application/vnd.github.v3.text-match+json`. Это уникальный заголовок `Accept` для GitHub. В данном случае содержимое представлено в специальном JSON формате.

# %% [markdown] id="riFAHLVUWMg0"
# **HTTP-методы в requests**

# %% [markdown] id="6edhr9EdWMdS"
# Помимо **GET**, в библиотеке реализованы и другие HTTP-методы, такие как **POST**, **PUT**, **DELETE**, **HEAD**, **PATCH** и **OPTIONS**. Для каждого из этих методов существует своя структура запросов, которая очень похожа на метод `get()`.

# %% colab={"base_uri": "https://localhost:8080/"} id="t9EZnmDDj45Q" outputId="3ae7300a-2d49-4169-c809-c49f0d45e75a"
requests.post('https://httpbin.org/post', data={'key':'value'})
requests.put('https://httpbin.org/put', data={'key':'value'})
requests.delete('https://httpbin.org/delete')
requests.head('https://httpbin.org/get')
requests.patch('https://httpbin.org/patch', data={'key':'value'})
requests.options('https://httpbin.org/get')

# %% [markdown] id="onAVc9Gtp3kF"
# Несмотря на отличия в HTTP-методах, схема работы с ответами будет аналогична, как мы работали с GET-запросом.

# %% [markdown] id="2hIJ49xTWMZ5"
# Каждая функция создает запрос к **httpbin.org** сервису, используя при этом ответный **HTTP-метод**. Это удобный сервис для тестирования REST API.
# Это чрезвычайно полезный сервис, созданный человеком, который внедрил использование `requests` – Кеннетом Рейтцом. Данный сервис предназначен для тестовых запросов. Здесь можно составить пробный запрос и получить ответ с требуемой информацией.

# %% colab={"base_uri": "https://localhost:8080/"} id="uCeojKOylQUV" outputId="18fb44e5-3885-499c-830b-47d4171821c5"
response = requests.head('https://httpbin.org/get')
print(f'Content-Type: {response.headers["Content-Type"]}')


response = requests.delete('https://httpbin.org/delete')
response.json()

# %% [markdown] id="-RQS0z2tWMTb"
# При использовании каждого из данных методов в объекте **Response** могут быть возвращены заголовки, тело запроса, коды состояния и многие другие аспекты.
#
# Методы **HEAD**, **PATCH** и **OPTIONS** не используются в **REST API**, поэтому мы не будем на них останавливаться.
#
#
#
#

# %% [markdown] id="q8fAcxX8WMQI"
# **Передача параметров через тело запроса**

# %% [markdown] id="y7o0jUe3WMMg"
# В соответствии со спецификацией HTTP запросы **POST**, **PUT** и **PATCH** передают информацию через тело запроса, а не через параметры строки запроса. Используя библиотеку `requests`, можно передать данные в параметр `data`.
#
# В свою очередь `data` использует словарь, список кортежей, байтов или объект файла. Это особенно важно, так как может возникнуть необходимость адаптации отправляемых с запросом данных в соответствии с определенными параметрами сервера.
#
# К примеру, если тип содержимого запроса `application/x-www-form-urlencoded`, следует отправлять данные формы в виде словаря.
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="cL6om38ZoCvb" outputId="1a089d16-87c5-43cc-d25a-34dc87f72fbe"
request = requests.post('https://httpbin.org/post', data={'key':'value'})
request.json()

# %% [markdown] id="-iifKUPOpEEx"
# Ту же самую информацию также можно отправить в виде списка кортежей:

# %% colab={"base_uri": "https://localhost:8080/"} id="d4mtUrRZorcg" outputId="6382a33d-3e4c-4903-e16c-5557acaccfb1"
request = requests.post('https://httpbin.org/post', data=[('key', 'value')])
request.json()

# %% [markdown] id="FBbwk-j7WMIB"
# В том случае, если требуется отравить данные **JSON**, можно использовать параметр `json`. При передачи данных **JSON** через `json`, `requests` произведет сериализацию данных и добавит правильный `Content-Type` заголовок.

# %% colab={"base_uri": "https://localhost:8080/"} id="akaZHLn3qhLd" outputId="a892b578-510d-470c-f6b8-ba1f9e3fccad"
response = requests.post('https://httpbin.org/post', json={'key':'value'})
json_response = response.json()
print(f'Отправленные данные: {json_response["data"]}')
print(f'Заголовок: {json_response["headers"]["Content-Type"]}')

# %% [markdown] id="LOqCyr1uqWsd"
# Мы видим, что сервер получил данные и HTTP заголовки, отправленные вместе с запросом. `requests` также предоставляет информацию в форме `PreparedRequest` (подготовленных к отправке данных).

# %% [markdown] id="ltpbskX5spSV"
# **PreparedRequest (подготовленных данных)**

# %% [markdown] id="71HtravvqXIX"
# При составлении запроса стоит иметь в виду, что перед его фактической отправкой на целевой сервер библиотека `requests` выполняет определенную подготовку. Подготовка запроса включает в себя такие вещи, как проверка заголовков и сериализация содержимого JSON.
#
# Если обратиться к атрибуту `request`, то можно просмотреть объект **PreparedRequest** (подготовленных данных):

# %% colab={"base_uri": "https://localhost:8080/"} id="j80xa-dZsUey" outputId="1502ce96-f4b4-46ce-c4ac-07fdd40124a7"
response = requests.post('https://httpbin.org/post', json={'key':'value'})

print('Подготовленные к отправке данные: ')
print('Content-Type: ', response.request.headers['Content-Type'])
print('URI: ', response.request.url)
print('Тело запроса: ', response.request.body)

# %% [markdown] id="h9BbhqieqXjs"
# Проверка **PreparedRequest** открывает доступ ко всей информации о выполняемом запросе. Это может быть пейлоад, URI, заголовки, аутентификация и многое другое.
#
# У всех описанных ранее типов запросов была одна общая черта – они представляли собой неаутентифицированные запросы к публичным API. Однако, подобающее большинство служб, с которыми может столкнуться пользователь, запрашивают аутентификацию.

# %% [markdown] id="Z2S19dvRuY3H"
# #### Аутентификация HTTP AUTH

# %% [markdown] id="x6F58f41ugOL"
# Аутентификация помогает сервису понять, кто вы. Как правило, вы предоставляете свои учетные данные на сервер, передавая данные через заголовок `Authorization` или пользовательский заголовок, определенной службы.
#
# Одним из примеров API, который требует аутентификации, является [Authenticated User API](https://docs.github.com/ru/rest/users?apiVersion=2022-11-28#get-the-authenticated-user) на GitHub. Это раздел веб-сервиса, который предоставляет информацию о профиле аутентифицированного пользователя. Чтобы отправить запрос API-интерфейсу аутентифицированного пользователя, вы можете передать свое имя пользователя и пароль на GitHub через кортеж в  методе `get()`.

# %% colab={"base_uri": "https://localhost:8080/"} id="AmBfFOd1viKE" outputId="be8caf3c-79bb-46ad-afe0-7fed5e130754"
from getpass import getpass # ввод пароля в колабе
requests.get('https://api.github.com/user', auth=('username', getpass()))

# %% [markdown] id="5dwSuitEuufA"
# Запрос будет выполнен успешно, если учетные данные, которые вы передали в кортеже `auth`, соответствуют реальному пользователю ГитХаба. Если выполнить запрос без учетных данных или с неверными данными, то получим код состояния **401 Unauthorized**.

# %% [markdown] id="emlpwKQmwf2m"
# Теперь, когда вы знаете о REST API практически все, пора [приступить](https://colab.research.google.com/drive/1_AzAVys4xub3yyw763NDwfeJ3WecGgkb) к самой главной части урока - создание своего собственного веб-сервиса на базе фреймворка FastAPI.
