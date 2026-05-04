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
        cleaned = name.strip()
        if len(cleaned) < 2 or not any(c.isalpha() for c in cleaned):
            return "Пожалуйста, скажите ваше имя."
        return self._save_name(cleaned)

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
            if 'error' in weather:
                self._set_state(DialogState.WAIT_CITY)
                return f"❌ Город \"{city}\" не найден. Проверьте написание и введите снова:"
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

        if 'error' in weather:
            # Город не найден — возвращаемся в WAIT_CITY чтобы пользователь мог исправить
            self._set_state(DialogState.WAIT_CITY)
            return f"❌ Город \"{city}\" не найден. Проверьте написание и введите снова:"

        result = weather_api.format_weather_message(weather)
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

    # ── Словарь словесных чисел ──────────────────────────────────────────────

    _WORD_NUMBERS = {
        "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3,
        "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
        "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
        "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
        "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
        "девятнадцать": 19, "двадцать": 20, "тридцать": 30,
        "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
        "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
        "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
        "пятьсот": 500, "шестьсот": 600, "семьсот": 700,
        "восемьсот": 800, "девятьсот": 900, "тысяча": 1000,
        # опечатки
        "всемь": 7, "осемь": 8, "восем": 8,
    }

    def _parse_number(self, token: str) -> float | None:
        """Пробует распарсить токен как число — цифровое или словесное."""
        token = token.strip().lower()
        try:
            return float(token)
        except ValueError:
            pass
        # составные: "двадцать пять" уже разбиты, но попробуем одиночные
        val = self._WORD_NUMBERS.get(token)
        return float(val) if val is not None else None

    def _extract_numbers(self, match) -> tuple[float, float] | None:
        """Извлекает два числа из match (группы 1 и 2)."""
        a = self._parse_number(match.group(1))
        b = self._parse_number(match.group(2))
        if a is None or b is None:
            return None
        return a, b

    def _fmt(self, n: float) -> str:
        """Форматирует число: убирает .0 у целых."""
        return str(int(n)) if n == int(n) else str(n)

    # ── Арифметика ────────────────────────────────────────────────────────────

    def handle_addition(self, match) -> str:
        nums = self._extract_numbers(match)
        if nums is None:
            return "Не могу распознать числа для сложения."
        a, b = nums
        res = a + b
        return f"{self._fmt(a)} + {self._fmt(b)} = {self._fmt(res)}"

    def handle_subtraction(self, match) -> str:
        nums = self._extract_numbers(match)
        if nums is None:
            return "Не могу распознать числа для вычитания."
        a, b = nums
        res = a - b
        return f"{self._fmt(a)} - {self._fmt(b)} = {self._fmt(res)}"

    def handle_multiplication(self, match) -> str:
        nums = self._extract_numbers(match)
        if nums is None:
            return "Не могу распознать числа для умножения."
        a, b = nums
        res = a * b
        return f"{self._fmt(a)} × {self._fmt(b)} = {self._fmt(res)}"

    def handle_division(self, match) -> str:
        nums = self._extract_numbers(match)
        if nums is None:
            return "Не могу распознать числа для деления."
        a, b = nums
        if b == 0:
            return "На ноль делить нельзя."
        res = a / b
        return f"{self._fmt(a)} ÷ {self._fmt(b)} = {self._fmt(res)}"

    def handle_power(self, match) -> str:
        nums = self._extract_numbers(match)
        if nums is None:
            return "Не могу распознать числа для возведения в степень."
        a, b = nums
        res = a ** b
        return f"{self._fmt(a)} ^ {self._fmt(b)} = {self._fmt(res)}"

    def handle_math_word(self, match) -> str:
        """Обрабатывает словесные выражения: 'восемь плюс три', 'сто минус сорок'."""
        import re as _re
        text = match.group(0).lower()

        # Убираем вводные слова
        text = _re.sub(r"^(сколько будет|вычисли|посчитай|сколько)\s+", "", text).strip()

        # Операторы — сначала многословные, потом короткие
        ops_ordered = [
            ("умножить на", "×"),
            ("разделить на", "÷"),
            ("делить на",   "÷"),
            ("делённое на", "÷"),
            ("в степени",   "^"),
            ("плюс",        "+"),
            ("прибавить",   "+"),
            ("сложить",     "+"),
            ("минус",       "-"),
            ("вычесть",     "-"),
            ("отнять",      "-"),
            ("умножить",    "×"),
            ("разделить",   "÷"),
            ("делить",      "÷"),
            ("степень",     "^"),
        ]

        op_found = op_sym = None
        for word, sym in ops_ordered:
            if word in text:
                op_found, op_sym = word, sym
                break

        if not op_found:
            return "Не могу определить операцию."

        parts = text.split(op_found, 1)
        if len(parts) != 2:
            return "Не могу распознать числа."

        left_tokens  = parts[0].strip().split()
        right_tokens = parts[1].strip().split()

        a = self._parse_word_sequence(left_tokens)
        b = self._parse_word_sequence(right_tokens)

        if a is None or b is None:
            return "Не могу распознать числа в выражении."

        try:
            if op_sym == "+":   res = a + b
            elif op_sym == "-": res = a - b
            elif op_sym == "×": res = a * b
            elif op_sym == "÷":
                if b == 0: return "На ноль делить нельзя."
                res = a / b
            elif op_sym == "^": res = a ** b
            else: return "Неизвестная операция."
            return f"{self._fmt(a)} {op_sym} {self._fmt(b)} = {self._fmt(res)}"
        except Exception:
            return "Не могу выполнить вычисление."

    def _parse_word_sequence(self, tokens: list) -> float | None:
        """Складывает последовательность словесных чисел: ['двадцать','пять'] → 25."""
        total = 0
        for t in tokens:
            v = self._parse_number(t)
            if v is None:
                return None
            total += v
        return float(total) if tokens else None

    # ── Остальное ─────────────────────────────────────────────────────────────

    # ── Анекдот ───────────────────────────────────────────────────────────────

    _JOKES = [
        "— Штирлиц выстрелил и промахнулся.\n— Промах, — подумал Штирлиц.\n— Штирлиц знает русский язык, — насторожился Мюллер.",
        "Штирлиц шёл по коридору. Навстречу шёл Мюллер.\n— Штирлиц! — окликнул его Мюллер.\nШтирлиц сделал вид, что не слышит.\n— Штирлиц! — снова крикнул Мюллер.\nШтирлиц сделал вид, что не слышит.\n— Штирлиц, вы глухой?\n— Нет.\n— Странно, — подумал Штирлиц.",
        "Вовочка принёс домой двойку.\n— Вовочка, что скажет папа, когда увидит дневник?\n— Ничего. Он у нас интеллигентный.",
        "— Вовочка, сколько будет дважды два?\n— Четыре.\n— Садись, пять.",
        "Учительница:\n— Вовочка, ты почему опять опоздал?\n— Я поскользнулся на льду.\n— И что, это заняло целый час?\n— Нет, но каждый раз, когда я вставал, — поскальзывался снова.",
        "— Рабинович, вы знаете четыре языка?\n— Ну, знаю.\n— И всё равно такой бедный?\n— А что, богатые знают пять?",
        "— Доктор, я буду жить?\n— А смысл?",
        "— Доктор, у меня склероз.\n— И давно?\n— Что давно?",
        "— Больной, вы принимали таблетки, которые я вам выписал?\n— Да, доктор, одну штуку.\n— Почему одну? Я же написал — по одной три раза в день!\n— Так я же ещё не три раза болел!",
        "Чукча пишет письмо в издательство:\n— Однако, получил вашу книгу. Однако, очень толстая. Однако, читал три дня. Однако, понравилась. Пришлите ещё одну такую же, однако.",
        "— Ты что, пьян?\n— Нет, я просто устал.\n— А от чего?\n— От вопросов.",
        "Муж возвращается домой раньше времени и застаёт жену с любовником.\n— Что происходит?! — кричит муж.\n— Видишь? — говорит жена любовнику. — Я же говорила, что он придёт пьяный!",
        "— Дорогой, ты меня любишь?\n— Да.\n— А докажи!\n— Я на тебе женился — это не доказательство?",
        "Новый русский приходит в библиотеку:\n— Дайте мне котлету.\n— Это библиотека!\n— (шёпотом) Дайте мне котлету.",
        "— Папа, купи собаку!\n— Нет.\n— Тогда кошку!\n— Нет.\n— Тогда рыбку!\n— Нет.\n— Тогда хомяка!\n— Нет.\n— Тогда муху!\n— …ладно.",
        "Штирлиц открыл дверь. Дверь закрылась. «Странно», — подумал Штирлиц и открыл снова. Дверь снова закрылась. Тут Штирлиц заметил, что дверь — на пружине. «Ещё страннее», — подумал Штирлиц.",
        "— Как дела?\n— Нормально.\n— А подробнее?\n— Нормально-нормально.",
        "Объявление в зоопарке: «Не кормите крокодила — он сытый и злой. Накормите — просто злой».",
        "— Рабинович, почему вы всегда отвечаете вопросом на вопрос?\n— А почему бы и нет?",
        "Идёт мужик по пустыне, навстречу верблюд.\n— Эй, верблюд, далеко до оазиса?\n— Два часа ходьбы.\n— Спасибо!\n— Не за что. Давно с людьми не разговаривал.",
        "— Алло, это психиатрическая больница?\n— Да.\n— А Иванов у вас лежит?\n— Сейчас проверим… Нет, такого нет.\n— Значит, я и правда галлюцинирую.",
        "Учитель:\n— Петров, назови три рода существительных!\n— Мужской, женский и средний.\n— Правильно. А теперь пример каждого.\n— Папа, мама и дитя.",
        "— Скажите, вы верите в жизнь после смерти?\n— Не знаю, но зарплату лучше платите до.",
        "Чиновник умер и попал на небо. Апостол Пётр:\n— Добро пожаловать. У нас тут рай и ад. Можете выбрать.\n— А взятки берёте?\n— Нет.\n— Тогда в ад.",
        "— Доктор, сколько мне жить?\n— Десять.\n— Чего десять? Лет? Месяцев?\n— …девять.",
        "Мальчик спрашивает отца:\n— Папа, откуда я взялся?\n— Тебя принёс аист.\n— А тебя?\n— И меня аист.\n— Получается, в нашей семье уже несколько поколений не было нормальных родов?",
        "— Официант, в моём супе муха!\n— Не кричите, а то все захотят.",
        "Объявление: «Продаётся шикарная вилла с видом на море».\nВнизу приписка: «Вид на море — отдельно».",
        "— Вы говорите по-английски?\n— Йес.\n— Как дела?\n— Йес.",
        "Штирлиц увидел свет в конце тоннеля. «Поезд», — подумал Штирлиц. Поезд тоже подумал.",
    ]

    def handle_joke(self, match=None) -> str:
        """Возвращает случайный анекдот с anekdot.ru, иначе — локальный список."""
        import random
        try:
            jokes = self._fetch_jokes_anekdot_ru()
            if jokes:
                return f"😄 {random.choice(jokes).strip()}"
        except Exception:
            pass
        return f"😄 {random.choice(self._JOKES)}"

    def _fetch_jokes_anekdot_ru(self) -> list[str]:
        """Парсит свежие анекдоты с нескольких RSS-лент anekdot.ru."""
        import urllib.request, re, html as html_module
        urls = [
            "https://www.anekdot.ru/rss/export_j.xml",
            "https://www.anekdot.ru/rss/export_s.xml",
        ]
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        jokes = []
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=6) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                items = re.findall(r"<description>(.*?)</description>", raw, re.DOTALL)
                for item in items[1:]:
                    text = item
                    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
                    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", "", text)
                    text = html_module.unescape(text)
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    if len(text) > 20:
                        jokes.append(text)
            except Exception:
                continue
        return jokes

    # ── Плейлист ──────────────────────────────────────────────────────────────

    def handle_playlist(self, match=None, original_text: str = "") -> str:
        """Составляет плейлист по настроению через SoundCloud."""
        try:
            from soundcloud_api import soundcloud_api
        except ImportError:
            return "❌ Файл soundcloud_api.py не найден. Убедись, что он лежит рядом с handlers.py."
        text = original_text or (match.group(0) if match else "")
        data = soundcloud_api.get_playlist(text, count=10)
        return soundcloud_api.format_playlist(data)

    def handle_how_are_you(self, match=None) -> str:
        return "У меня всё отлично, спасибо!"

    def handle_time(self, match=None) -> str:
        return f"Сейчас {datetime.now().strftime('%H:%M')}"


    def handle_smalltalk(self, match=None) -> str:
        import random
        responses = [
            "Я чат-бот, умею: узнавать погоду, говорить время, поддерживать беседу. Попробуйте спросить 'какая погода?' или 'который час?'",
            "Я умею рассказывать погоду, сообщать время и просто общаться. Чем могу помочь?",
            "Меня создали чтобы помогать с погодой, временем и общением. Спросите что-нибудь!",
            "Я бот-помощник. Могу рассказать погоду в любом городе, сообщить текущее время или просто поговорить.",
            "Мои навыки: погода 🌤, время 🕐, общение 💬. Что вас интересует?",
        ]
        return random.choice(responses)

    def handle_unknown(self, match=None) -> str:
        state = self.state
        if state == DialogState.WAIT_NAME:
            return "Пожалуйста, скажите ваше имя."
        if state == DialogState.WAIT_CITY:
            return "Я всё ещё жду название города. Так где смотрим погоду?"
        if state == DialogState.WAIT_DATE:
            return "Укажите дату: сегодня, завтра или послезавтра."
        return "Я не понимаю этот запрос. Попробуйте спросить о погоде, времени или поздороваться."