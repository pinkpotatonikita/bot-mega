import os
import spacy
from spacy.matcher import Matcher
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

# ── BERT-классификатор (Fine-Tuned RuBERT) ────────────────────────────────────
# Загружается один раз при импорте. Если модель не обучена — is_ready=False,
# и classify_intent автоматически использует Word Embeddings как fallback.
try:
    from bert_classifier import bert_classifier as _bert
except ImportError:
    _bert = None


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
        Загружает обучающую выборку из intents.csv (рядом с этим файлом).
        Если файл не найден — падает с понятной ошибкой.
        """
        import csv
        csv_path = os.path.join(os.path.dirname(__file__), "intents.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Файл {csv_path} не найден. "
                "Положите intents.csv рядом с nlp_processor.py"
            )
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [(row["text"], row["intent"]) for row in reader]

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
        Определяет интент текста.

        Стратегия (двухуровневый fallback):
          1. Fine-Tuned RuBERT (bert_classifier) — если модель обучена и готова.
          2. Word Embeddings + LogisticRegression (spaCy) — если BERT недоступен.

        threshold применяется только для Word Embeddings (BERT всегда возвращает argmax).
        """
        cleaned = text.strip()
        if len(cleaned) < 2:
            return None

        # ── Уровень 1: Fine-Tuned BERT ────────────────────────────────────────
        if _bert is not None and _bert.is_ready:
            result = _bert.predict(cleaned)
            if result:
                return result

        # ── Уровень 2: Word Embeddings fallback ───────────────────────────────
        if not self.intent_classifier:
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