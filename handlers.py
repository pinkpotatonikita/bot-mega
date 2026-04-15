from datetime import datetime
from nlp_processor import NLPProcessor
from weather_api import weather_api
from dialog_manager import dialog_manager, DialogState


# Временный ID гостя — используется пока пользователь не назвал имя.
# Позволяет хранить состояние WAIT_NAME до создания реального ID.
_GUEST_ID = 0


class Handlers:
    def __init__(self):
        self.name:            str | None = None
        self.current_user_id: int        = _GUEST_ID  # всегда int, никогда None
        self.nlp = NLPProcessor()

    # ── Вспомогательные свойства ──────────────────────────────────────────────

    @property
    def state(self) -> DialogState:
        return dialog_manager.get_state(self.current_user_id)

    def _set_state(self, state: DialogState) -> None:
        dialog_manager.set_state(self.current_user_id, state)

    def _reset(self) -> None:
        dialog_manager.reset(self.current_user_id)

    # ── NLP ───────────────────────────────────────────────────────────────────

    def process_with_nlp(self, message: str) -> dict | None:
        return self.nlp.analyze_text(message)

    # ── Приветствие / прощание ────────────────────────────────────────────────

    def handle_greeting(self, match=None) -> str:
        if self.name:
            return f"Здравствуйте, {self.name}! Чем могу помочь?"
        # Имя неизвестно — переходим в WAIT_NAME через гостевой ID
        self._set_state(DialogState.WAIT_NAME)
        return "Здравствуйте! Как вас зовут?"

    def handle_farewell(self, match=None) -> str:
        self._reset()
        return f"До свидания, {self.name}!" if self.name else "До свидания!"

    # ── Имя ───────────────────────────────────────────────────────────────────

    def handle_set_name(self, match) -> str:
        """Вызывается regex-паттерном 'меня зовут X'."""
        return self._save_name(match.group(1).capitalize())

    def handle_name_response(self, name: str) -> str:
        """Вызывается когда пользователь ответил на вопрос 'Как вас зовут?'."""
        return self._save_name(name)

    def _save_name(self, name: str) -> str:
        dialog_manager.reset(_GUEST_ID)          # чистим гостевое состояние
        self.name = name
        self.current_user_id = hash(name) % (10 ** 8)
        dialog_manager.reset(self.current_user_id)
        return f"Приятно познакомиться, {self.name}!"

    # ── Погода ────────────────────────────────────────────────────────────────

    def handle_weather(self, match=None, nlp_analysis=None) -> str:
        if not self.name:
            return "Сначала представьтесь, пожалуйста!"

        city = None
        if match and match.lastindex and match.group(1):
            city = match.group(1).strip()
        elif nlp_analysis and nlp_analysis.get('city'):
            city = nlp_analysis['city']

        if city:
            self._reset()
            weather = weather_api.get_weather_direct(city)
            return weather_api.format_weather_message(weather)

        self._set_state(DialogState.WAIT_CITY)
        return "В каком городе вас интересует погода?"

    def handle_city_response(self, city: str) -> str:
        """Вызывается когда пользователь ответил на вопрос 'В каком городе?'."""
        dialog_manager.set_data(self.current_user_id, 'city', city)

        # ── Доп. задание: спрашиваем дату ────────────────────────────────────
        self._set_state(DialogState.WAIT_DATE)
        return f"На какую дату вас интересует погода в городе {city}?"

        # ── Базовое задание (без WAIT_DATE): раскомментируйте строки ниже,
        # закомментируйте две строки выше ─────────────────────────────────────
        # self._reset()
        # weather = weather_api.get_weather_direct(city)
        # return weather_api.format_weather_message(weather)

    def handle_date_response(self, date_text: str) -> str:
        """Вызывается когда пользователь ответил на вопрос 'На какую дату?'."""
        city = dialog_manager.get_data(self.current_user_id).get('city', '')
        self._reset()

        if not city:
            return "Что-то пошло не так. Попробуйте снова: спросите погоду."

        offset, label = self._parse_date_offset(date_text)
        weather = weather_api.get_weather_direct(city)
        result  = weather_api.format_weather_message(weather)
        return result if offset == 0 else f"Прогноз на {label}:\n{result}"

    @staticmethod
    def _parse_date_offset(text: str) -> tuple[int, str]:
        text = text.lower()
        for word, value in {
            "сегодня":       (0, "сегодня"),
            "завтра":        (1, "завтра"),
            "послезавтра":   (2, "послезавтра"),
            "через два дня": (2, "послезавтра"),
            "через 2 дня":   (2, "послезавтра"),
        }.items():
            if word in text:
                return value
        return 0, "сегодня"

    # ── Математика ────────────────────────────────────────────────────────────

    def handle_addition(self, match) -> str:
        try:
            a, b = float(match.group(1)), float(match.group(2))
            return f"Результат: {a} + {b} = {a + b}"
        except Exception:
            return "Не могу выполнить сложение"

    def handle_subtraction(self, match) -> str:
        try:
            a, b = float(match.group(1)), float(match.group(2))
            return f"Результат: {a} - {b} = {a - b}"
        except Exception:
            return "Не могу выполнить вычитание"

    # ── Остальное ─────────────────────────────────────────────────────────────

    def handle_how_are_you(self, match=None) -> str:
        return "У меня всё отлично, спасибо!"

    def handle_time(self, match=None) -> str:
        return f"Сейчас {datetime.now().strftime('%H:%M')}"

    def handle_unknown(self, match=None) -> str:
        state = self.state
        if state == DialogState.WAIT_NAME:
            return "Пожалуйста, скажите ваше имя."
        if state == DialogState.WAIT_CITY:
            return "Я всё ещё жду название города. Так где смотрим погоду?"
        if state == DialogState.WAIT_DATE:
            return "Укажите дату: сегодня, завтра или послезавтра."
        return "Я не понимаю этот запрос. Попробуйте спросить о погоде, времени или поздороваться."