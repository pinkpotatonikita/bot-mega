import requests
from datetime import datetime, timedelta
import re
import spacy

# Используем md-модель — она содержит векторы и лучше лемматизирует
# Если md нет — fallback на sm
try:
    nlp = spacy.load("ru_core_news_md")
except OSError:
    nlp = spacy.load("ru_core_news_sm")


def _nlp_lemmatize_city(city_text: str) -> str:
    """
    Лемматизирует название города через spaCy.
    Возвращает строку с леммами токенов, каждая с заглавной буквы.
    Например: «Арзамасе» → «Арзамас», «Нижнем Новгороде» → «Нижний Новгород»
    """
    doc = nlp(city_text.strip())
    return " ".join(token.lemma_.capitalize() for token in doc if not token.is_punct)


class WeatherAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)

    # ──────────────────────────────────────────────────────────────────────────
    # Извлечение города из произвольного текста
    # ──────────────────────────────────────────────────────────────────────────

    def extract_city_from_text(self, text: str) -> str | None:
        """
        Извлекает и нормализует (→ именительный падеж) название города из текста.

        Стратегии (в порядке приоритета):
          1. NER (GPE/LOC) — берём лемму корня сущности.
          2. Синтаксис: существительное/собственное имя, зависящее от предлога «в».
          3. Regex «в/во <Заглавное слово>» + лемматизация каждого слова через spaCy.

        Работает для любых городов — как известных spaCy, так и малоизвестных,
        потому что лемматизация опирается на морфологию, а не на словарь имён.
        """
        doc = nlp(text)
        print(f"  Анализирую: '{text}'")

        # ── 1. NER ────────────────────────────────────────────────────────────
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                # Берём лемму синтаксического корня сущности (правильный падеж)
                lemma = ent.root.lemma_.capitalize()
                # Для многословных топонимов восстанавливаем все слова
                if len(ent) > 1:
                    lemma = " ".join(t.lemma_.capitalize() for t in ent)
                print(f"  ✓ NER ({ent.label_}): '{ent.text}' → '{lemma}'")
                return lemma

        # ── 2. Синтаксис: предлог «в/во» → его дополнение ────────────────────
        for token in doc:
            if token.lower_ in ("в", "во") and token.dep_ == "case":
                head = token.head
                # Собираем всю именную группу вокруг head
                noun_phrase_tokens = [
                    t for t in head.subtree
                    if t.pos_ in ("NOUN", "PROPN", "ADJ") and t.dep_ != "case"
                ]
                if noun_phrase_tokens:
                    lemma = " ".join(t.lemma_.capitalize() for t in noun_phrase_tokens)
                    print(f"  ✓ Синтаксис (case): '{head.text}' → '{lemma}'")
                    return lemma

        # Второй проход: ищем предлог «в» как prep-зависимость у глагола
        for token in doc:
            if token.lower_ in ("в", "во") and token.pos_ == "ADP":
                for child in token.head.children:
                    if child.dep_ in ("obl", "nmod", "nsubj") and child.pos_ in ("NOUN", "PROPN"):
                        lemma = child.lemma_.capitalize()
                        print(f"  ✓ Синтаксис (obl): '{child.text}' → '{lemma}'")
                        return lemma

        # ── 3. Regex + лемматизация ───────────────────────────────────────────
        # Ищем заглавное слово после «в/во» (предложный падеж)
        match = re.search(r"\b[вВ][оО]?\s+([А-ЯЁ][а-яёА-ЯЁ\s\-]{1,30})", text)
        if match:
            raw = match.group(1).strip()
            # Берём только первую «именную» часть до любого глагола/союза
            raw_doc = nlp(raw)
            city_tokens = []
            for t in raw_doc:
                if t.pos_ in ("NOUN", "PROPN", "ADJ"):
                    city_tokens.append(t.lemma_.capitalize())
                else:
                    break
            if city_tokens:
                lemma = " ".join(city_tokens)
                print(f"  ✓ Regex+NLP: '{raw}' → '{lemma}'")
                return lemma

        print(f"  ✗ Город не найден")
        return None

    def _normalize_city(self, city: str) -> str:
        """
        Нормализует название города, переданного напрямую (не из предложения).
        Если город уже в именительном падеже — spaCy вернёт его без изменений.
        Если в косвенном — вернёт лемму.
        """
        city = re.sub(r'^[вВ][оО]?\s+', '', city.strip())
        return _nlp_lemmatize_city(city)

    # ──────────────────────────────────────────────────────────────────────────
    # Запросы к API
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_weather_for_city(self, city: str) -> dict:
        """Делает запрос к OpenWeatherMap для конкретного города."""
        cache_key = city.lower()
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data

        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric',
                'lang': 'ru',
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather_info = {
                    'city': data['name'],
                    'requested_city': city,
                    'country': data['sys']['country'],
                    'temperature': round(data['main']['temp']),
                    'feels_like': round(data['main']['feels_like']),
                    'description': data['weather'][0]['description'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'pressure': data['main']['pressure'],
                }
                self.cache[cache_key] = (weather_info, datetime.now())
                return weather_info
            else:
                return {'error': f'Город "{city}" не найден (HTTP {response.status_code})'}
        except Exception as e:
            return {'error': f'Ошибка соединения: {str(e)}'}

    def get_weather_direct(self, city: str) -> dict:
        """
        Получает погоду по готовому названию города (пользователь ввёл его напрямую,
        например в режиме ожидания города).

        Нормализует через spaCy (один вызов — без ручных таблиц окончаний),
        затем делает запрос. Если нормализованный вариант не найден — пробует
        исходный (на случай если spaCy вернул артефакт).
        """
        city = re.sub(r'^[вВ][оО]?\s+', '', city.strip())
        print(f"🔍 Прямой запрос погоды для города: '{city}'")

        normalized = _nlp_lemmatize_city(city)
        print(f"  NLP-нормализация: '{city}' → '{normalized}'")

        # Пробуем нормализованный вариант
        if normalized.lower() != city.lower():
            result = self._fetch_weather_for_city(normalized)
            if 'error' not in result:
                return result

        # Fallback: исходный вариант (вдруг spaCy что-то исказил)
        result = self._fetch_weather_for_city(city)
        return result

    def get_weather(self, user_input: str) -> dict:
        """Получает погоду из произвольного пользовательского ввода (с NLP-разбором)."""
        city = self.extract_city_from_text(user_input)
        if not city:
            return {'error': 'Не могу найти название города в запросе'}
        print(f"🔍 Город из запроса: '{city}'")
        return self._fetch_weather_for_city(city)

    # ──────────────────────────────────────────────────────────────────────────
    # Форматирование
    # ──────────────────────────────────────────────────────────────────────────

    def format_weather_message(self, weather_data: dict) -> str:
        """Форматирует данные погоды в читаемое сообщение."""
        if 'error' in weather_data:
            return f"❌ {weather_data['error']}"

        emoji_map = {
            'ясно': '☀️', 'солнечно': '☀️',
            'облачно': '☁️', 'пасмурно': '☁️',
            'дождь': '🌧️', 'снег': '❄️',
            'гроза': '⛈️', 'туман': '🌫️',
        }
        desc = weather_data['description'].lower()
        emoji = next((v for k, v in emoji_map.items() if k in desc), '🌡️')

        return (
            f"{emoji} Погода в {weather_data['city']}, {weather_data['country']}:\n"
            f"🌡️ Температура: {weather_data['temperature']}°C"
            f" (ощущается как {weather_data['feels_like']}°C)\n"
            f"📝 {weather_data['description'].capitalize()}\n"
            f"💧 Влажность: {weather_data['humidity']}%\n"
            f"💨 Ветер: {weather_data['wind_speed']} м/с\n"
            f"📊 Давление: {weather_data['pressure']} гПа"
        )


API_KEY = "e1eceb3fe5be1742bc305b2b4a02b6a8"
weather_api = WeatherAPI(API_KEY)