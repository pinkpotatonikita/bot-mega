"""
train_bert.py
=============
Fine-tuning DeepPavlov/rubert-base-cased для классификации интентов чат-бота.

Пайплайн:
    intents.csv
        → токенизация (AutoTokenizer)
        → IntentDataset (torch.utils.data.Dataset)
        → AutoModelForSequenceClassification
        → Trainer API (HuggingFace)
        → сохранение в ./intent_model/

Использование:
    pip install transformers datasets torch scikit-learn pandas
    python train_bert.py

После обучения модель используется в bert_classifier.py.
"""

import os
import json

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── Константы ─────────────────────────────────────────────────────────────────

MODEL_NAME  = "DeepPavlov/rubert-base-cased"
DATA_PATH   = os.path.join(os.path.dirname(__file__), "intents.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "intent_model")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MAX_LEN    = 64
BATCH_SIZE = 8
EPOCHS     = 4
LR         = 2e-5
TEST_SIZE  = 0.15
SEED       = 42

# ── Dataset ───────────────────────────────────────────────────────────────────

class IntentDataset(torch.utils.data.Dataset):
    """PyTorch Dataset для токенизированных интентов."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── Вспомогательные функции ───────────────────────────────────────────────────

def tokenize(tokenizer: AutoTokenizer, texts: list[str]) -> dict:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# ── Главная функция обучения ──────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("Fine-Tuning RuBERT для классификации интентов")
    print("=" * 60)

    # ── 1. Загрузка данных ────────────────────────────────────────────────────
    print(f"\n[1/6] Загружаем датасет: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"      Всего примеров: {len(df)}")
    print(f"      Интенты: {sorted(df['intent'].unique())}")

    # ── 2. Кодирование меток ──────────────────────────────────────────────────
    print("\n[2/6] Кодируем метки...")
    label2id = {label: idx for idx, label in enumerate(sorted(df["intent"].unique()))}
    id2label  = {v: k for k, v in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    # Сохраняем маппинг меток рядом с моделью
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"      label2id: {label2id}")

    # ── 3. Разбивка на train/val ──────────────────────────────────────────────
    print(f"\n[3/6] Разбиваем: train={1-TEST_SIZE:.0%} / val={TEST_SIZE:.0%}")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )
    print(f"      Train: {len(train_texts)}, Val: {len(val_texts)}")

    # ── 4. Токенизация ────────────────────────────────────────────────────────
    print(f"\n[4/6] Загружаем токенизатор: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenize(tokenizer, train_texts)
    val_encodings   = tokenize(tokenizer, val_texts)

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset   = IntentDataset(val_encodings,   val_labels)

    # ── 5. Модель ─────────────────────────────────────────────────────────────
    print(f"\n[5/6] Загружаем модель: {MODEL_NAME} ({len(label2id)} классов)")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # ── 6. Обучение ───────────────────────────────────────────────────────────
    print(f"\n[6/6] Запускаем обучение ({EPOCHS} эпохи, batch={BATCH_SIZE}, lr={LR})...")

    training_args = TrainingArguments(
        output_dir                  = RESULTS_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        logging_steps               = 10,
        learning_rate               = LR,
        seed                        = SEED,
        report_to                   = "none",   # отключаем wandb/tensorboard
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
    )

    trainer.train()

    # ── Итоговая оценка ───────────────────────────────────────────────────────
    print("\n📊 Оценка на val-выборке:")
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(axis=-1)
    print(classification_report(val_labels, preds,
                                 target_names=list(label2id.keys()),
                                 zero_division=0))

    # ── Сохранение ────────────────────────────────────────────────────────────
    print(f"\n💾 Сохраняем модель в: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Готово! Модель сохранена.")
    print(f"   Для загрузки в боте используйте bert_classifier.py")


if __name__ == "__main__":
    train()