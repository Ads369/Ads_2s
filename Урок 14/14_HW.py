# %%  Cell 1: Imports and Initialize
import ast
import asyncio
import os
import re
from typing import Optional

import mwclient
import mwparserfromhell
import pandas as pd
import tiktoken
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message
from openai import OpenAI
from scipy import spatial

# %%  Cell 1: Constants
# Constanats
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
SAVE_PATH = "./winter_olympics_2022.csv"
GLOBAL_DATA = None
SYSTEM_MESSAGE = "You answer questions about the 2022 Winter Olympics."
TOPIC = "2022 Winter Olympics"
CATEGORY_TITLE = f"Category:{TOPIC}"
WIKI_SITE = "en.wikipedia.org"
SECTIONS_TO_IGNORE = set(
    [
        "See also",
        "References",
        "External links",
        "Further reading",
        "Footnotes",
        "Bibliography",
        "Sources",
        "Citations",
        "Literature",
        "Footnotes",
        "Notes and references",
        "Photo gallery",
        "Works cited",
        "Photos",
        "Gallery",
        "Notes",
        "References and sources",
        "References and notes",
    ]
)


def setup_secret_key(key_name: str) -> Optional[str]:
    """
    Set up the OpenAI API key, either from environment or user input.
    """
    _key_name = os.environ.get(key_name)

    if _key_name:
        print(f"{key_name} is already set in the environment.")
        return _key_name

    try:
        from dotenv import load_dotenv

        load_dotenv()
        _key_name = os.environ.get(f"{key_name}")
        if _key_name:
            print(f"{key_name} loaded from .env file.")
            return _key_name
    except ImportError:
        print("Warning: dotenv package not found. Unable to load from .env file.")

    try:
        import getpass

        _key_name = getpass.getpass(f"Enter {key_name}: ")
        os.environ[f"{key_name}"] = _key_name
        print(f"{key_name} set from user input.")
        return _key_name
    except Exception as e:
        print(f"Error setting API key: {e}")
        return None


OPENAI_KEY = setup_secret_key("OPENAI_API_KEY")
OPENAI_CLIENT = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
if OPENAI_CLIENT is not None:
    print("OpenAI is Ready")

TELEGRAM_BOT_TOKEN = setup_secret_key("TELEGRAM_BOT_TOKEN")
if TELEGRAM_BOT_TOKEN is None:
    raise ValueError("Telegram bot token is not set.")
bot = Bot(TELEGRAM_BOT_TOKEN)
if  bot is not None:
    print("Telegram bot is Ready")
dp = Dispatcher()
router = Router()

# %% Cell 3: Fix All errors
import nest_asyncio
nest_asyncio.apply()



# %%  Cell 3: Utile functions
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    # Функция возвращает число токенов в строке для заданной модели
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def clip_tokens(text: str, length: int = 4096, model: str = GPT_MODEL) -> str:
    # Функция обрезает строку до заданного числа токенов, учитывая модель
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text=text)

    token_value = [
        encoding.decode_single_token_bytes(token).decode("utf-8", errors="ignore")
        for token in tokens[0:length]
    ]
    return "".join(token_value)


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str]:
    """Разделяет строку надвое с помощью разделителя (delimiter),
    пытаясь сбалансировать токены с каждой стороны.
    """

    # Делим строку на части по разделителю, по умолчанию \n - перенос строки
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # разделитель не найден
    elif len(chunks) == 2:
        return chunks  # нет необходимости искать промежуточную точку
    else:
        # Считаем токены
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        # Предварительное разделение по середине числа токенов
        best_diff = halfway
        # В цикле ищем какой из разделителей, будет ближе всего к best_diff
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff

        # TODO Fix errors
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        # Возвращаем левую и правую часть оптимально разделенной строки
        return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Обрезка строки до максимально разрешенного числа токенов.

    Args:
        string (str): Строка.
        model (str): Модель токенизации.
        max_tokens (int): Максимальное число разрешенных токенов.
        print_warning (bool, optional): Флаг вывода предупреждения. Defaults to True.

    Returns:
        str: Обрезанная строка.
    """
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    # Обрезаем строку и декодируем обратно
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(
            f"Предупреждение: Строка обрезана с {len(encoded_string)} токенов до {max_tokens} токенов."
        )
    # Усеченная строка
    return truncated_string


def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Разделяет секции на список из частей секций, в каждой части не более max_tokens.
    Каждая часть представляет собой кортеж родительских заголовков [H1, H2, ...] и текста (str).

    Args:
        subsection (tuple[list[str], str]): Кортеж с родительскими заголовками и текстом.
        model (str, optional): Модель токенизации. Defaults to GPT_MODEL.
        max_recursion (int, optional): Максимальное число рекурсий. Defaults to 5.

    Returns:
        list[str]: Список строк, разделенных на части секции.
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # Если длина соответствует допустимой, то вернет строку
    if num_tokens_in_string <= max_tokens:
        return [string]
    # если в результате рекурсия не удалось разделить строку, то просто усечем ее по числу токенов
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # иначе разделим пополам и выполним рекурсию
    else:
        titles, text = subsection
        for delimiter in [
            "\n\n",
            "\n",
            ". ",
        ]:  # Пробуем использовать разделители от большего к меньшему (разрыв, абзац, точка)
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # если какая-либо половина пуста, повторяем попытку с более простым разделителем
                continue
            else:
                # применим рекурсию на каждой половине
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion
                        - 1,  # уменьшаем максимальное число рекурсий
                    )
                    results.extend(half_strings)
                return results
    # иначе никакого разделения найдено не было, поэтому просто обрезаем строку (должно быть очень редко)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]


# %%  Markdown cells:
# Алгоритм обучения Search-Ask
# %%  Cell 5: Data functions
def get_preset_data():
    embeddings_path = (
        "https://storage.yandexcloud.net/academy.ai/winter_olympics_2022.csv"
    )
    df = pd.read_csv(embeddings_path)
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    return df


GLOBAL_DATA = get_preset_data()


# %%  Cell 6
# Функция поиска
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> tuple[list[str], list[float]]:
    """Возвращает отранжированные строки и их релевантность запросу

    Args:
        query (str): пользовательский запрос
        df (pd.DataFrame): DataFrame со столбцами text и embedding (база знаний)
        relatedness_fn (callable, optional): функция схожести, по умолчанию косинусное расстояние
        top_n (int, optional): выбор лучших n-результатов. По умолчанию 100.

    Returns:
        tuple[list[str], list[float]]: кортеж из двух списков - строки и их релевантность
    """
    query_embedding_response = OPENAI_CLIENT.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    # Получен токенизированный пользовательский запрос
    query_embedding = query_embedding_response.data[0].embedding

    # Сравниваем пользовательский запрос с каждой токенизированной строкой DataFrame
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]

    # Сортируем по убыванию схожести полученный список
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)

    # Преобразовываем наш список в кортеж из списков
    strings, relatednesses = zip(*strings_and_relatednesses)

    # Возвращаем n лучших результатов
    return strings[:top_n], relatednesses[:top_n]


# %%  Cell
# Функция формирования запроса к chatGPT по пользовательскому вопросу и базе знаний
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int,
) -> str:
    """Возвращает сообщение для GPT с соответствующими исходными текстами,
    извлеченными из фрейма данных (базы знаний).

    Args:
        query (str): пользовательский запрос
        df (pd.DataFrame): DataFrame со столбцами text и embedding (база знаний)
        model (str): модель
        token_budget (int): ограничение на число отсылаемых токенов в модель

    Returns:
        str: сообщение для GPT с соответствующими исходными текстами,
        извлеченными из
    """

    # ранжирования базы знаний по пользовательскому запросу
    strings, relatednesses = strings_ranked_by_relatedness(query, df)

    # Шаблон инструкции для chatGPT
    message = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'

    # Шаблон для вопроса
    question = f"\n\nQuestion: {query}"

    # Добавляем к сообщению для chatGPT релевантные строки из базы знаний, пока не выйдем за допустимое число токенов
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question


# %%  Cell
def ask(
    query: str,
    df: pd.DataFrame | None = None,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str | None:
    """Отвечает на вопрос, используя GPT и базу знаний.

    Args:
        query (str): пользовательский запрос
        df (pd.DataFrame, optional): DataFrame со столбцами text и embedding (база знаний). Defaults to df.
        model (str, optional): модель.
        token_budget (int, optional): ограничение на число отсылаемых токенов в модель.
        print_message (bool, optional): нужно ли выводить сообщение перед отправкой.

    Returns:
        str: ответ на вопрос
    """

    if df is None:
        if GLOBAL_DATA is None:
            df = get_preset_data()
        else:
            df = GLOBAL_DATA
    else:
        raise ValueError("df is not None")

    message = query_message(query, df, model=model, token_budget=token_budget)

    if print_message:
        print(message)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {"role": "user", "content": message},
    ]
    response = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=0,  # гиперпараметр степени случайности при генерации текста.
        # Влияет на то, как модель выбирает следующее слово в последовательности.
    )
    response_message = response.choices[0].message.content
    return response_message


# %% Cell: Test  ask()
# ask('How many records were set at the 2022 Winter Olympics?')
# ask("Did Jamaica or Cuba have more athletes at the 2022 Winter Olympics?")


# %% Cell: search info
# Соберем заголовки всех статей
def titles_from_category(
    category: mwclient.listing.Category,
    max_depth: int,  # type: ignore
) -> set[str]:
    """Возвращает набор заголовков страниц в данной категории Википедии и ее подкатегориях.

    Args:
        category (mwclient.listing.Category): Категория статей Википедии.
        max_depth (int): Глубина вложения статей.

    Returns:
        set[str]: Набор заголовков страниц в данной категории Википедии и ее подкатегориях.
    """
    titles = set()
    for cm in category.members():  # Перебираем вложенные объекты категории
        if isinstance(cm, mwclient.page.Page):  # type: ignore
            titles.add(cm.name)
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:  # type: ignore
            # Если объект является категорией и глубина вложения не достигла максимальной
            # вызываем рекурсивно функцию для подкатегории
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)
            titles.update(deeper_titles)
    return titles


# Инициализация объекта MediaWiki
site = mwclient.Site(WIKI_SITE)
category_page = site.pages[CATEGORY_TITLE]
# Получение множества всех заголовков категории с вложенностью на один уровень
titles = titles_from_category(category_page, max_depth=1)
print(f"Создано {len(titles)} заголовков статей в категории {CATEGORY_TITLE}.")


# %% Cell
# Функция возвращает список всех вложенных секций для заданной секции страницы Википедии
def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode,  # текущая секция
    parent_titles: list[str],  # Заголовки родителя
    sections_to_ignore: set[str],  # Секции, которые необходимо проигнорировать
) -> list[tuple[list[str], str]]:
    """
    Из раздела Википедии возвращает список всех вложенных секций.
    Каждый подраздел представляет собой кортеж, где:
      - первый элемент представляет собой список родительских секций, начиная с заголовка страницы
      - второй элемент представляет собой текст секции
    """

    # Извлекаем заголовки текущей секции
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    # Заголовки Википедии имеют вид: "== Heading =="

    if title.strip("=" + " ") in sections_to_ignore:
        # Если заголовок секции в списке для игнора, то пропускаем его
        return []

    # Объединим заголовки и подзаголовки, чтобы сохранить контекст для chatGPT
    titles = parent_titles + [title]

    # Преобразуем wikicode секции в строку
    full_text = str(section)

    # Выделяем текст секции без заголовка
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        # Если один заголовок, то формируем результирующий список
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        # Формируем результирующий список из текста до первого подзаголовка
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(
                # Вызываем функцию получения вложенных секций для заданной секции
                all_subsections_from_section(subsection, titles, sections_to_ignore)
            )  # Объединяем результирующие списки данной функции и вызываемой
        return results


# Функция возвращает список всех секций страницы, за исключением тех, которые отбрасываем
def all_subsections_from_title(
    title: str,  # Заголовок статьи Википедии, которую парсим
    sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,  # Секции, которые игнорируем
    site_name: str = WIKI_SITE,  # Ссылка на сайт википедии
) -> list[tuple[list[str], str]]:
    """
    Из заголовка страницы Википедии возвращает список всех вложенных секций.
    Каждый подраздел представляет собой кортеж, где:
      - первый элемент представляет собой список родительских секций, начиная с заголовка страницы
      - второй элемент представляет собой текст секции
    """

    # Инициализация объекта MediaWiki
    # WIKI_SITE ссылается на англоязычную часть Википедии
    site = mwclient.Site(site_name)

    # Запрашиваем страницу по заголовку
    page = site.pages[title]

    # Получаем текстовое представление страницы
    text = page.text()

    # Удобный парсер для MediaWiki
    parsed_text = mwparserfromhell.parse(text)
    # Извлекаем заголовки
    headings = [str(h) for h in parsed_text.filter_headings()]
    if headings:  # Если заголовки найдены
        # В качестве резюме берем текст до первого заголовка
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        # Если нет заголовков, то весь текст считаем резюме
        summary_text = str(parsed_text)
    results = [([title], summary_text)]  # Добавляем резюме в результирующий список
    for subsection in parsed_text.get_sections(
        levels=[2]
    ):  # Извлекаем секции 2-го уровня
        results.extend(
            # Вызываем функцию получения вложенных секций для заданной секции
            all_subsections_from_section(subsection, [title], sections_to_ignore)
        )  # Объединяем результирующие списки данной функции и вызываемой
    return results


# Разбивка статей на секции
# придется немного подождать, так как на парсинг 100 статей требуется около минуты
wikipedia_sections = []
for title in titles:
    wikipedia_sections.extend(all_subsections_from_title(title))
print(f"Найдено {len(wikipedia_sections)} секций на {len(titles)} страницах")


# %% cell: clear
# Очистка текста секции от ссылок <ref>xyz</ref>, начальных и конечных пробелов
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    titles, text = section
    # Удаляем ссылки
    text = re.sub(r"<ref.*?</ref>", "", text)
    # Удаляем пробелы вначале и конце
    text = text.strip()
    return (titles, text)


# Применим функцию очистки ко всем секциям с помощью генератора списков
wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]


# Отфильтруем короткие и пустые секции
def keep_section(section: tuple[list[str], str]) -> bool:
    """Возвращает значение True, если раздел должен быть сохранен, в противном случае значение False."""
    titles, text = section
    # Фильтруем по произвольной длине, можно выбрать и другое значение
    if len(text) < 16:
        return False
    else:
        return True


original_num_sections = len(wikipedia_sections)
wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
print(
    f"Отфильтровано {original_num_sections-len(wikipedia_sections)} секций, осталось {len(wikipedia_sections)} секций."
)
# for ws in wikipedia_sections[:5]:
#     print(ws[0])
#     print(ws[1][:50] + "...")
#     print()

# %% Cell:  test
# Делим секции на части
MAX_TOKENS = 1600
wikipedia_strings = []
for section in wikipedia_sections:
    wikipedia_strings.extend(
        split_strings_from_subsection(section, max_tokens=MAX_TOKENS)
    )

print(
    f"{len(wikipedia_sections)} секций Википедии поделены на {len(wikipedia_strings)} строк."
)


# %% Cell: tokening
# Функция отправки chatGPT строки для ее токенизации (вычисления эмбедингов)
def get_embedding(text, model=EMBEDDING_MODEL):
    return OPENAI_CLIENT.embeddings.create(input=[text], model=model).data[0].embedding


def save_embeddings(path=SAVE_PATH):
    df = pd.DataFrame({"text": wikipedia_strings[:10]})
    df["embedding"] = df.text.apply(lambda x: get_embedding(x, model=EMBEDDING_MODEL))
    df.to_csv(path, index=False)
    return df


GLOBAL_DATA = save_embeddings()


def calculate_lines_csv():
    df = GLOBAL_DATA
    if df is None:
        df = get_preset_data()
    return len(df.text.str.split("\n").apply(len).unique())


# %% Cell: Telegram bot


@router.message(Command("start", "help"))
async def send_welcome(message: Message):
    await message.reply(
        f"Hi! I'm ChatGPT-Winter-Olympics-Bot by @Ads_2s"
        f"I can answer questions about the {TOPIC}"
        f"I have {calculate_lines_csv()} lines in my database."
        "---"
        f"Send me any message and I'll try to answer it."
        f"Example: Did Jamaica or Cuba have more athletes at the 2022 Winter Olympics?"
    )


@router.message()
async def gpt_answer(message: Message):
    if message is None:
        await message.answer(message.text)


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
