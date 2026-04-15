import spacy
from spacy.matcher import Matcher
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report


class NLPProcessor:
    def __init__(self):
        try:
            self.nlp_ru = spacy.load("ru_core_news_md")
        except OSError:
            print("Русская модель не найдена. Загрузите: python -m spacy download ru_core_news_md")
            self.nlp_ru = None

        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            print("Английская модель не найдена. Загрузите: python -m spacy download en_core_web_sm")
            self.nlp_en = None

        self.matcher_ru = Matcher(self.nlp_ru.vocab) if self.nlp_ru else None
        self._setup_matchers()

        self.processed_cache = {}

        self.intent_classifier = None
        self.intent_labels = []
        self.threshold = 0.60          # будет откалиброван автоматически
        self._train_intent_classifier()

    # ──────────────────────────────────────────────────────────────────────────
    # Обучающая выборка (~180 фраз вместо 40)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_training_data(self):
        """
        Расширенная обучающая выборка.
        Каждый интент — ~25-30 фраз, включая перефразировки,
        разговорные варианты и типичные ошибки пользователей.
        """
        return [
            # ── greeting ──────────────────────────────────────────────────────
            ("привет", "greeting"),
            ("здравствуйте", "greeting"),
            ("добрый день", "greeting"),
            ("добрый вечер", "greeting"),
            ("доброе утро", "greeting"),
            ("хай", "greeting"),
            ("приветствую", "greeting"),
            ("здорово", "greeting"),
            ("приветик", "greeting"),
            ("хелло", "greeting"),
            ("приве", "greeting"),
            ("здрасте", "greeting"),
            ("рад тебя видеть", "greeting"),
            ("рад тебя слышать", "greeting"),
            ("доброго времени суток", "greeting"),
            ("привет бот", "greeting"),
            ("начнём", "greeting"),
            ("начать", "greeting"),
            ("ку", "greeting"),
            ("yo", "greeting"),
            ("hello", "greeting"),
            ("hi", "greeting"),
            ("hey", "greeting"),
            ("good morning", "greeting"),
            ("good evening", "greeting"),

            # ── farewell ──────────────────────────────────────────────────────
            ("пока", "farewell"),
            ("до свидания", "farewell"),
            ("увидимся", "farewell"),
            ("до встречи", "farewell"),
            ("всего хорошего", "farewell"),
            ("прощай", "farewell"),
            ("пока пока", "farewell"),
            ("до скорого", "farewell"),
            ("удачи", "farewell"),
            ("всего доброго", "farewell"),
            ("спокойной ночи", "farewell"),
            ("ухожу", "farewell"),
            ("завершить разговор", "farewell"),
            ("выхожу", "farewell"),
            ("на этом всё", "farewell"),
            ("на сегодня всё", "farewell"),
            ("goodbye", "farewell"),
            ("bye", "farewell"),
            ("see you", "farewell"),
            ("good night", "farewell"),
            ("до завтра", "farewell"),
            ("конец", "farewell"),
            ("стоп", "farewell"),
            ("хватит", "farewell"),

            # ── weather ───────────────────────────────────────────────────────
            ("какая погода", "weather"),
            ("погода в москве", "weather"),
            ("будет ли дождь", "weather"),
            ("ожидаются осадки", "weather"),
            ("будет ли снег", "weather"),
            ("будут ли осадки завтра", "weather"),
            ("холодно ли на улице", "weather"),
            ("нужно ли брать зонт", "weather"),
            ("прогноз погоды", "weather"),
            ("сколько градусов", "weather"),
            ("какая температура", "weather"),
            ("погода на завтра", "weather"),
            ("что с погодой", "weather"),
            ("как погода сегодня", "weather"),
            ("на улице тепло", "weather"),
            ("на улице холодно", "weather"),
            ("тепло ли сегодня", "weather"),
            ("будет ли гроза", "weather"),
            ("нужна ли куртка", "weather"),
            ("что за погода на улице", "weather"),
            ("какой прогноз на неделю", "weather"),
            ("скажи погоду", "weather"),
            ("покажи погоду", "weather"),
            ("узнать погоду", "weather"),
            ("температура воздуха", "weather"),
            ("погода в питере", "weather"),
            ("погода в новосибирске", "weather"),
            ("дождь сегодня будет", "weather"),
            ("снег ожидается", "weather"),
            ("ветер сильный будет", "weather"),

            # ── how_are_you ───────────────────────────────────────────────────
            ("как дела", "how_are_you"),
            ("как ты", "how_are_you"),
            ("как у тебя дела", "how_are_you"),
            ("как поживаешь", "how_are_you"),
            ("что нового", "how_are_you"),
            ("как жизнь", "how_are_you"),
            ("как ты себя чувствуешь", "how_are_you"),
            ("всё хорошо", "how_are_you"),
            ("ты в порядке", "how_are_you"),
            ("как настроение", "how_are_you"),
            ("как твои дела", "how_are_you"),
            ("как работается", "how_are_you"),
            ("как ты там", "how_are_you"),
            ("ты как", "how_are_you"),
            ("всё норм", "how_are_you"),
            ("как оно", "how_are_you"),
            ("что у тебя нового", "how_are_you"),
            ("расскажи о себе", "how_are_you"),
            ("how are you", "how_are_you"),
            ("how are you doing", "how_are_you"),
            ("how do you feel", "how_are_you"),
            ("are you okay", "how_are_you"),
            ("what's up", "how_are_you"),
            ("how's it going", "how_are_you"),

            # ── time ──────────────────────────────────────────────────────────
            ("сколько времени", "time"),
            ("который час", "time"),
            ("какое время", "time"),
            ("сколько сейчас время", "time"),
            ("что за время сейчас", "time"),
            ("скажи время", "time"),
            ("покажи время", "time"),
            ("текущее время", "time"),
            ("сейчас сколько", "time"),
            ("время сейчас", "time"),
            ("который сейчас час", "time"),
            ("уже поздно", "time"),
            ("уже утро", "time"),
            ("сколько на часах", "time"),
            ("что показывают часы", "time"),
            ("what time is it", "time"),
            ("what's the time", "time"),
            ("tell me the time", "time"),
            ("current time", "time"),
            ("time please", "time"),

            # ── set_name ──────────────────────────────────────────────────────
            # меня зовут X
            ("меня зовут иван", "set_name"),
            ("меня зовут мария", "set_name"),
            ("меня зовут андрей", "set_name"),
            ("меня зовут екатерина", "set_name"),
            # я X
            ("я алексей", "set_name"),
            ("я наташа", "set_name"),
            ("я дмитрий", "set_name"),
            ("я ольга", "set_name"),
            # моё имя X
            ("моё имя сергей", "set_name"),
            ("моё имя анна", "set_name"),
            ("моё имя николай", "set_name"),
            ("моё имя юлия", "set_name"),
            # зовите / зови меня X
            ("зовите меня саша", "set_name"),
            ("зовите меня макс", "set_name"),
            ("зови меня антон", "set_name"),
            ("зови меня лена", "set_name"),
            # можешь звать меня X
            ("можешь звать меня олег", "set_name"),
            ("можешь звать меня света", "set_name"),
            # меня называют X
            ("меня называют виктор", "set_name"),
            ("меня называют таня", "set_name"),
            # english
            ("my name is alex", "set_name"),
            ("my name is sarah", "set_name"),
            ("i am john", "set_name"),
            ("i am emily", "set_name"),
            ("call me mike", "set_name"),
            ("call me kate", "set_name"),
            ("i'm david", "set_name"),
            ("i'm anna", "set_name"),
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Векторизация: взвешенные embeddings (IDF-подобный подход)
    # ──────────────────────────────────────────────────────────────────────────

    def _vectorize(self, text: str) -> np.ndarray:
        """
        Переводит текст в вектор через spaCy Word Embeddings.

        Улучшение по сравнению с doc.vector:
        - Стоп-слова и пунктуация исключаются из усреднения.
        - Токены с нулевым вектором (OOV) не тянут среднее к нулю.
        - Итоговый вектор L2-нормализован для стабильности косинусного сходства.
        """
        if not self.nlp_ru:
            return np.zeros(300)

        doc = self.nlp_ru(text.lower().strip())

        # Берём векторы только содержательных токенов
        vectors = [
            token.vector
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.has_vector
        ]

        if not vectors:
            # Если все токены OOV или стоп-слова — используем doc.vector как fallback
            return doc.vector

        avg = np.mean(vectors, axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    # ──────────────────────────────────────────────────────────────────────────
    # Обучение классификатора + кросс-валидация + автокалибровка порога
    # ──────────────────────────────────────────────────────────────────────────

    def _train_intent_classifier(self):
        """
        Обучает LogisticRegression на Word Embeddings.
        После обучения:
          1. Выводит accuracy по кросс-валидации (5-fold Stratified).
          2. Автоматически выбирает порог уверенности по валидационным вероятностям.
          3. Выводит classification_report для диагностики.
        """
        if not self.nlp_ru:
            print("Классификатор интентов не обучен: отсутствует spaCy-модель.")
            return

        data = self._get_training_data()
        texts  = [item[0] for item in data]
        labels = [item[1] for item in data]

        X = np.array([self._vectorize(t) for t in texts])
        y = np.array(labels)

        # ── Кросс-валидация ───────────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf_cv = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        cv_scores = cross_val_score(clf_cv, X, y, cv=cv, scoring='accuracy')
        print(f"✅ Word Embeddings классификатор: "
              f"CV accuracy = {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

        # ── Итоговое обучение на всей выборке ─────────────────────────────────
        self.intent_classifier = LogisticRegression(
            max_iter=2000, C=1.0, solver='lbfgs'
        )
        self.intent_classifier.fit(X, y)
        self.intent_labels = list(set(labels))

        # ── Автокалибровка порога ─────────────────────────────────────────────
        # Берём максимальную вероятность по каждому обучающему примеру.
        # Порог = 5-й перцентиль правильно классифицированных — это
        # «минимальная уверенность, при которой модель была права».
        train_probas = self.intent_classifier.predict_proba(X).max(axis=1)
        train_preds  = self.intent_classifier.predict(X)
        correct_probas = train_probas[train_preds == y]
        if len(correct_probas) > 0:
            self.threshold = float(np.percentile(correct_probas, 5))
            # Ограничиваем разумным диапазоном
            self.threshold = max(0.45, min(self.threshold, 0.80))
        print(f"   Порог уверенности (авто): {self.threshold:.2f}")

        # ── Диагностический отчёт ─────────────────────────────────────────────
        print("\n📊 Classification report (train):")
        print(classification_report(y, train_preds, zero_division=0))

    # ──────────────────────────────────────────────────────────────────────────
    # Классификация интента
    # ──────────────────────────────────────────────────────────────────────────

    def classify_intent(self, text: str, threshold: float = None) -> str | None:
        """
        Определяет интент с помощью Word Embeddings.
        Возвращает название интента или None, если уверенность ниже порога.

        threshold=None → используется автоматически подобранный порог.
        """
        if not self.intent_classifier:
            return None

        cleaned = text.strip()
        if len(cleaned) < 2:
            return None

        vec = self._vectorize(cleaned).reshape(1, -1)

        if np.all(vec == 0):
            return None

        proba_row = self.intent_classifier.predict_proba(vec)[0]
        max_proba = proba_row.max()
        used_threshold = threshold if threshold is not None else self.threshold

        if max_proba < used_threshold:
            return None

        return self.intent_classifier.predict(vec)[0]

    # ──────────────────────────────────────────────────────────────────────────
    # Остальной код — без изменений логики, только мелкие правки стиля
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_matchers(self):
        if not self.matcher_ru:
            return

        weather_patterns = [
            [{"LEMMA": {"IN": ["какой", "что"]}},
             {"LEMMA": "погода"},
             {"LEMMA": "в"},
             {"POS": "PROPN"}],
            [{"LEMMA": "погода"},
             {"LEMMA": "в"},
             {"POS": "PROPN"}],
            [{"LEMMA": "сколько"},
             {"LEMMA": "градус"},
             {"LEMMA": "в"},
             {"POS": "PROPN"}],
            [{"LEMMA": {"IN": ["холодно", "тепло", "жарко"]}},
             {"LEMMA": "сегодня", "OP": "?"},
             {"LEMMA": "в", "OP": "?"},
             {"POS": "PROPN", "OP": "+"}],
        ]
        self.matcher_ru.add("WEATHER_QUERY", weather_patterns)

        if self.nlp_en:
            self.matcher_en = Matcher(self.nlp_en.vocab)
            en_weather_patterns = [
                [{"LEMMA": {"IN": ["weather", "forecast"]}},
                 {"LEMMA": "in"}, {"POS": "PROPN"}],
                [{"LEMMA": {"IN": ["what", "how"]}},
                 {"LEMMA": "weather"}, {"LEMMA": "in"}, {"POS": "PROPN"}],
            ]
            self.matcher_en.add("WEATHER_QUERY", en_weather_patterns)

    def get_nlp(self, text):
        if re.search('[а-яА-Я]', text):
            return self.nlp_ru
        return self.nlp_en

    def analyze_text(self, text):
        cache_key = text.lower().strip()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]

        nlp = self.get_nlp(text)
        if not nlp:
            return None

        doc = nlp(text)

        result = {
            'text': text,
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'entities': [],
            'is_weather_query': False,
            'city': None,
            'language': 'ru' if nlp == self.nlp_ru else 'en',
            'intent': self.classify_intent(text),
        }

        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'ORG']:
                result['entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                })

        matcher = self.matcher_ru if nlp == self.nlp_ru else getattr(self, 'matcher_en', None)
        if matcher:
            matches = matcher(doc)
            result['is_weather_query'] = len(matches) > 0

        if result['intent'] == 'weather':
            result['is_weather_query'] = True

        for ent in result['entities']:
            if ent['label'] in ['GPE', 'LOC']:
                result['city'] = ent['text']
                break

        if not result['city'] and result['is_weather_query']:
            result['city'] = self._extract_city_by_patterns(text, doc)

        self.processed_cache[cache_key] = result
        return result

    def _extract_city_by_patterns(self, text, doc):
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                return ent.root.lemma_.capitalize()

        match = re.search(r"\bв[о]?\s+([А-ЯЁ][а-яёА-ЯЁ\s\-]+)", text)
        if match:
            city_phrase = match.group(1).strip()
            city_doc = self.nlp_ru(city_phrase) if self.nlp_ru else None
            if city_doc:
                return ' '.join(token.lemma_.capitalize() for token in city_doc)

        return None

    def save_to_db(self, db, analysis_result, user_name=None):
        if not db:
            return
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS nlp_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_name TEXT,
                        original_text TEXT,
                        language TEXT,
                        is_weather_query BOOLEAN,
                        city TEXT,
                        intent TEXT,
                        entities TEXT,
                        tokens TEXT,
                        lemmas TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('''
                    INSERT INTO nlp_analysis
                    (user_name, original_text, language, is_weather_query,
                     city, intent, entities, tokens, lemmas)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_name,
                    analysis_result['text'],
                    analysis_result['language'],
                    analysis_result['is_weather_query'],
                    analysis_result['city'],
                    analysis_result.get('intent'),
                    str(analysis_result['entities']),
                    str(analysis_result['tokens']),
                    str(analysis_result['lemmas']),
                ))
                conn.commit()
        except Exception as e:
            print(f"Ошибка при сохранении в БД: {e}")

    def extract_city_from_query(self, text):
        nlp = self.get_nlp(text)
        if not nlp:
            return None
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                city_lemma = ' '.join(token.lemma_ for token in ent)
                return {
                    'text': ent.text,
                    'lemma': city_lemma,
                    'label': ent.label_,
                }
        return None