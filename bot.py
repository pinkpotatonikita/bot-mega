import re
import string
from patterns import patterns
from handlers import Handlers
from logger import log_message, get_db
from weather_api import weather_api
from dialog_manager import DialogState

# Фразы, которые точно не являются названием города или датой
NON_CITY_PATTERNS = re.compile(
    r"^(привет|здравствуй|пока|до свидания|как дела|который час|"
    r"сколько времени|помоги|выход|exit|quit)$",
    re.IGNORECASE,
)


class ChatBot:
    def __init__(self):
        self.handlers = Handlers()
        self.patterns = patterns
        self.db = get_db()

        self.handler_map = {
            "greeting":    self.handlers.handle_greeting,
            "farewell":    self.handlers.handle_farewell,
            "set_name":    self.handlers.handle_set_name,
            "weather":     self.handlers.handle_weather,
            "addition":    self.handlers.handle_addition,
            "subtraction": self.handlers.handle_subtraction,
            "how_are_you": self.handlers.handle_how_are_you,
            "time":        self.handlers.handle_time,
        }

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def _preprocess(self, message: str) -> str:
        msg = message.lower().strip()
        return msg.translate(str.maketrans('', '', string.punctuation))

    def _extract_name(self, message: str) -> str | None:
        name_patterns = [
            r"меня зовут\s+([а-яА-Яa-zA-Z]+)",
            r"меня\s+([а-яА-Яa-zA-Z]+)\s+зовут",
            r"моё имя\s+([а-яА-Яa-zA-Z]+)",
            r"зови(?:те)? меня\s+([а-яА-Яa-zA-Z]+)",
            r"я\s+([а-яА-Яa-zA-Z]+)",
            r"name is\s+([а-яА-Яa-zA-Z]+)",
            r"i(?:'m| am)\s+([а-яА-Яa-zA-Z]+)",
        ]
        for pat in name_patterns:
            m = re.search(pat, message, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        # Одиночное слово — считаем именем
        words = message.strip().split()
        if len(words) == 1 and words[0].isalpha() and len(words[0]) > 1:
            return words[0].capitalize()
        return None

    # ── Главный метод обработки ───────────────────────────────────────────────

    def process(self, message: str) -> str:
        original  = message
        processed = self._preprocess(message)
        state     = self.handlers.state

        # ══════════════════════════════════════════════════════════════════════
        # FSM: обработка по текущему состоянию
        # ══════════════════════════════════════════════════════════════════════

        # ── WAIT_NAME: ждём имя ───────────────────────────────────────────────
        if state == DialogState.WAIT_NAME:
            name = self._extract_name(original)
            if name:
                return self.handlers.handle_name_response(name)
            return self.handlers.handle_unknown()

        # ── WAIT_CITY: ждём город ─────────────────────────────────────────────
        if state == DialogState.WAIT_CITY:
            # Если пользователь написал что-то явно не про город — сбрасываем
            if NON_CITY_PATTERNS.match(processed):
                self.handlers._reset()
                # Обрабатываем как обычное сообщение (fallthrough ниже)
            else:
                city = re.sub(r'^[вВ][оО]?\s+', '', original.strip())
                return self.handlers.handle_city_response(city)

        # ── WAIT_DATE: ждём дату ──────────────────────────────────────────────
        if state == DialogState.WAIT_DATE:
            return self.handlers.handle_date_response(original)

        # ══════════════════════════════════════════════════════════════════════
        # START: стандартный маршрут — NLP → regex
        # ══════════════════════════════════════════════════════════════════════

        # ── NLP-анализ ────────────────────────────────────────────────────────
        nlp_analysis = self.handlers.process_with_nlp(original)
        intent = nlp_analysis.get('intent') if nlp_analysis else None

        # ── Word Embeddings: используем интент если уверены ──────────────────
        if intent and intent != 'set_name':
            handler = self.handler_map.get(intent)
            if handler:
                if intent == 'weather':
                    return handler(match=None, nlp_analysis=nlp_analysis)
                return handler(match=None)

        # ── Regex-паттерны ────────────────────────────────────────────────────
        for pattern, handler_key in self.patterns:
            match = pattern.search(processed) or pattern.search(original)
            if match:
                handler = self.handler_map.get(handler_key)
                if handler_key == 'weather':
                    return handler(match=match, nlp_analysis=nlp_analysis)
                return handler(match)

        return self.handlers.handle_unknown()


# ── Статистика ────────────────────────────────────────────────────────────────

def show_stats(bot: ChatBot) -> None:
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ПОЛЬЗОВАТЕЛЕЙ")
    print("=" * 50)
    for user in bot.db.get_user_stats():
        print(f"Пользователь: {user['name']}")
        print(f"  Первое появление:          {user['first_seen']}")
        print(f"  Последнее взаимодействие:  {user['last_interaction']}")
        print(f"  Количество сообщений:      {user['total_messages']}")
        print(f"  Количество сессий:         {user['interactions_count']}")
        print("-" * 30)
    print("\nПОСЛЕДНИЕ СООБЩЕНИЯ:")
    print("-" * 30)
    for log in bot.db.get_recent_logs(5):
        user_name = log['user_name'] or 'Аноним'
        print(f"[{log['timestamp']}] {user_name}: {log['user_message'][:40]}...")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    bot = ChatBot()
    print("Чат-бот запущен (FSM + Word Embeddings + NLP). Введите 'пока' для выхода.")
    print("Команды: /stats — показать статистику")
    print("\nПримеры:")
    print("  'Какая погода?'          → бот спросит город, затем дату")
    print("  'Какая погода в Москве?' → сразу покажет")
    print("  'Будут ли осадки завтра?' → Word Embeddings распознает")
    print("-" * 50)

    while True:
        user_input = input("Вы: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ('пока', 'до свидания', 'exit', 'quit'):
            response = bot.handlers.handle_farewell()
            print(f"Бот: {response}")
            log_message(user_input, response, bot.handlers.name)
            break

        if user_input.lower() == '/stats':
            show_stats(bot)
            continue

        response = bot.process(user_input)
        print(f"Бот: {response}")
        log_message(user_input, response, bot.handlers.name)


if __name__ == "__main__":
    main()