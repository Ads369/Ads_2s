from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from service import (
    get_last_result,
    get_question,
    get_question_db,
    get_quiz_index,
    new_quiz,
    update_last_result,
    update_quiz_index,
)

router = Router()


@router.callback_query(F.data == "right_answer")
async def right_answer(callback: types.CallbackQuery):
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None,
    )
    quiz_data = await get_question_db(callback.message)

    await callback.message.answer("Верно!")
    current_question_index = await get_quiz_index(callback.from_user.id)
    # Обновление номера текущего вопроса в базе данных
    current_question_index += 1
    await update_quiz_index(callback.from_user.id, current_question_index)

    # Обновление результата в базе данных
    _result = await get_last_result(callback.from_user.id)
    await update_last_result(callback.from_user.id, int(_result) + 1)

    if current_question_index < len(quiz_data):
        await get_question(callback.message, callback.from_user.id)
    else:
        await callback.message.answer("Это был последний вопрос. Квиз завершен!")


@router.callback_query(F.data == "wrong_answer")
async def wrong_answer(callback: types.CallbackQuery):
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None,
    )
    quiz_data = await get_question_db(callback.message)

    # Получение текущего вопроса из словаря состояний пользователя
    current_question_index = await get_quiz_index(callback.from_user.id)
    correct_option = quiz_data[current_question_index]["correct_option"]

    await callback.message.answer(
        f"Неправильно. Правильный ответ: {quiz_data[current_question_index]['options'][correct_option]}"
    )

    # Обновление номера текущего вопроса в базе данных
    current_question_index += 1
    await update_quiz_index(callback.from_user.id, current_question_index)

    if current_question_index < len(quiz_data):
        await get_question(callback.message, callback.from_user.id)
    else:
        await callback.message.answer("Это был последний вопрос. Квиз завершен!")


# Хэндлер на команду /start
@router.message(Command("start"))
async def cmd_start(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Начать игру"))
    await message.answer_photo(
        photo="https://storage.yandexcloud.net/quiz-ads-backet/be902a6445867dff21399b8684d9c735.jpg",
        caption="Добро пожаловать в квиз!",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )



# Хэндлер на команду /start
@router.message(Command("help"))
async def cmd_help(message: types.Message):
    _result = await get_last_result(message.from_user.id)
    await message.answer("Ваш результат последнего квиза: {}/9".format(_result))


# Хэндлер на команду /quiz
@router.message(F.text == "Начать игру")
@router.message(Command("quiz"))
async def cmd_quiz(message: types.Message):
    await message.answer("Давайте начнем квиз!")
    await new_quiz(message)
