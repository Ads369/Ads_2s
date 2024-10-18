import json

from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from database import execute_select_query, execute_update_query, pool


def generate_options_keyboard(answer_options, right_answer):
    builder = InlineKeyboardBuilder()

    for option in answer_options:
        builder.add(
            types.InlineKeyboardButton(
                text=option,
                callback_data="right_answer"
                if option == right_answer
                else "wrong_answer",
            )
        )

    builder.adjust(1)
    return builder.as_markup()


async def get_question(message, user_id):
    _quiz_data = await get_question_db(message)

    # Получение текущего вопроса из словаря состояний пользователя
    current_question_index = await get_quiz_index(user_id)
    print(current_question_index)
    correct_index = _quiz_data[current_question_index]["correct_option"]

    opts = _quiz_data[current_question_index]["options"]

    kb = generate_options_keyboard(opts, opts[correct_index])
    await message.answer(
        f"{_quiz_data[current_question_index]['question']}", reply_markup=kb
    )


async def get_question_db(message):
    get_quiz_question = """
        SELECT *
        FROM `quiz_data_table`
    """
    _results = execute_select_query(pool, get_quiz_question)

    results = []
    for result in _results:
        result["options"] = json.loads(result["options"])
        results.append(result)

    return results


async def get_quiz_index_db(user_id):
    get_user_index = """
        DECLARE $user_id AS Uint64;
        SELECT question_index
        FROM `quiz_state`
        WHERE user_id == $user_id;
    """
    results = execute_select_query(pool, get_user_index, user_id=user_id)

    if len(results) == 0:
        return 0
    if results[0]["question_index"] is None:
        return 0
    return results[0]["question_index"]


async def new_quiz(message):
    user_id = message.from_user.id
    current_question_index = 0
    await update_quiz_index(user_id, current_question_index)
    await get_question(message, user_id)


async def get_quiz_index(user_id):
    get_user_index = """
        DECLARE $user_id AS Uint64;
        SELECT question_index
        FROM `quiz_state`
        WHERE user_id == $user_id;
    """
    results = execute_select_query(pool, get_user_index, user_id=user_id)

    if len(results) == 0:
        return 0
    if results[0]["question_index"] is None:
        return 0
    return results[0]["question_index"]


async def update_quiz_index(user_id, question_index):
    set_quiz_state = """
        DECLARE $user_id AS Uint64;
        DECLARE $question_index AS Uint64;

        UPSERT INTO `quiz_state` (`user_id`, `question_index`)
        VALUES ($user_id, $question_index);
    """

    execute_update_query(
        pool,
        set_quiz_state,
        user_id=user_id,
        question_index=question_index,
    )


async def get_last_result(user_id):
    get_last_result = """
        DECLARE $user_id AS Uint64;
        SELECT last_question_result
        FROM `quiz_result`
        WHERE user_id == $user_id;
    """
    results = execute_select_query(pool, get_last_result, user_id=user_id)

    if len(results) == 0:
        return 0
    if results[0]["last_question_result"] is None:
        return 0
    return results[0]["last_question_result"]


async def update_last_result(user_id, last_question_result):
    set_last_result = """
        DECLARE $user_id AS Uint64;
        DECLARE $last_question_result AS Uint64;

        UPSERT INTO `quiz_result` (`user_id`, `last_question_result`)
        VALUES ($user_id, $last_question_result);
    """

    execute_update_query(
        pool,
        set_last_result,
        user_id=user_id,
        last_question_result=last_question_result,
    )
