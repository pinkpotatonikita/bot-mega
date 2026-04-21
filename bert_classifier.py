"""
bert_classifier.py
==================
Модуль инференса для Fine-Tuned RuBERT.

Загружается один раз при старте бота.
Предоставляет функцию predict_intent(text) → str | None.

Требования:
    - Предварительно обученная модель в директории ./intent_model/
      (запустить train_bert.py перед использованием бота)
    - pip install transformers torch

Архитектура:
    Текст → AutoTokenizer → DeepPavlov/rubert-base-cased fine-tuned
          → logits → argmax → id2label → intent str
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Путь к сохранённой модели ──────────────────────────────────────────────────

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "intent_model")


class BERTIntentClassifier:
    """
    Обёртка над fine-tuned RuBERT для определения интента.

    Использование:
        classifier = BERTIntentClassifier()
        intent = classifier.predict("какая погода в москве")
        # → "weather"
    """

    def __init__(self, model_dir: str = _MODEL_DIR):
        self.model_dir   = model_dir
        self.tokenizer   = None
        self.model       = None
        self.id2label    = {}
        self.is_ready    = False
        self._load()

    # ── Загрузка ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        label_map_path = os.path.join(self.model_dir, "label_map.json")

        if not os.path.isdir(self.model_dir) or not os.path.exists(label_map_path):
            print(
                "⚠️  BERT-модель не найдена. Сначала запустите:\n"
                "       python train_bert.py\n"
                "   Бот продолжит работу с Word Embeddings (spaCy)."
            )
            return

        try:
            print(f"🤖 Загружаем BERT-модель из: {self.model_dir}")

            # Маппинг id → label
            with open(label_map_path, encoding="utf-8") as f:
                mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}

            # Токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            )
            self.model.eval()

            self.is_ready = True
            print(f"   ✅ BERT готов. Интенты: {list(self.id2label.values())}")

        except Exception as e:
            print(f"❌ Ошибка загрузки BERT: {e}\n   Используем fallback (spaCy).")
            self.is_ready = False

    # ── Инференс ──────────────────────────────────────────────────────────────

    def predict(self, text: str, threshold: float = 0.6) -> str | None:
        """
        Определяет интент текста.

        Возвращает строку-интент или None если модель не загружена
        или уверенность ниже threshold.
        Никогда не бросает исключения — только логирует.
        """
        if not self.is_ready:
            return None

        if not text or len(text.strip()) < 4:
            return None

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs           = torch.softmax(outputs.logits, dim=1)
            max_prob        = probs.max().item()

            if max_prob < threshold:
                return None

            predicted_class = probs.argmax(dim=1).item()
            return self.id2label.get(predicted_class)

        except Exception as e:
            print(f"⚠️  Ошибка BERT inference: {e}")
            return None

    # ── Пакетный инференс ─────────────────────────────────────────────────────

    def predict_batch(self, texts: list[str], threshold: float = 0.6) -> list[str | None]:
        """Определяет интенты для списка текстов (эффективнее поштучного)."""
        if not self.is_ready or not texts:
            return [None] * len(texts)

        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            results = []
            for i in range(len(texts)):
                max_prob = probs[i].max().item()
                if max_prob < threshold:
                    results.append(None)
                else:
                    results.append(self.id2label.get(probs[i].argmax().item()))
            return results

        except Exception as e:
            print(f"⚠️  Ошибка BERT batch inference: {e}")
            return [None] * len(texts)


# ── Глобальный синглтон ────────────────────────────────────────────────────────
# Импортируется в nlp_processor.py и bot.py

bert_classifier = BERTIntentClassifier()