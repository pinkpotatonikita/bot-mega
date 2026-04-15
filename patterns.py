import re

patterns = [
    # Приветствие — расширенный список включая сокращения
    (re.compile(
        r"привет|прив|приве|здравствуй|здравствуйте|здрасте|здорово|"
        r"добрый день|добрый вечер|доброе утро|доброго времени|хай|хелло|hello|hi\b|hey\b",
        re.IGNORECASE), "greeting"),

    # Прощание
    (re.compile(
        r"пока|до свидания|до встречи|до скорого|прощай|увидимся|"
        r"всего хорошего|всего доброго|спокойной ночи|bye\b|goodbye",
        re.IGNORECASE), "farewell"),

    # Имя
    (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), "set_name"),

    # Погода
    (re.compile(r"погода(?: в ([а-яА-Яa-zA-Z\- ]+))?", re.IGNORECASE), "weather"),

    # Математика
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), "addition"),
    (re.compile(r"(\d+)\s*-\s*(\d+)"), "subtraction"),

    # Как дела
    (re.compile(r"как (у тебя )?дела|как ты|как поживаешь|как жизнь", re.IGNORECASE), "how_are_you"),

    # Время
    (re.compile(r"(сколько|какое) (времени|время)|который час", re.IGNORECASE), "time"),
]