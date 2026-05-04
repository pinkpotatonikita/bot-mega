"""
Модуль голосового ввода для чат-бота Димсн.
Использует OpenAI Whisper для распознавания речи (оффлайн).

Защита от самозаписи:
  - tts.py вызывает set_speaking(True/False)
  - record_audio() активно ждёт в цикле пока флаг не снят + пауза после

Управление записью:
  - Запись идёт max seconds секунд
  - Enter досрочно останавливает запись
  - Если после Enter введён текст — он используется вместо голоса
"""

import re
import time
import threading
import logging
import sys

logger = logging.getLogger(__name__)

# ── Флаг: бот сейчас воспроизводит TTS ───────────────────────────────────────
_speaking_lock = threading.Lock()
_is_speaking = False


def set_speaking(active: bool) -> None:
    global _is_speaking
    with _speaking_lock:
        _is_speaking = active


def is_bot_speaking() -> bool:
    with _speaking_lock:
        return _is_speaking


# ── Загрузка модели ───────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def preload_model(model_name: str = "small"):
    global _model
    with _model_lock:
        if _model is None:
            try:
                import whisper
                print(f"[Voice] Загружаю Whisper ({model_name})...")
                _model = whisper.load_model(model_name)
                print("[Voice] Модель загружена ✓")
            except ImportError:
                logger.error("openai-whisper не установлен: pip install openai-whisper")
                raise
    return _model


def _get_model(model_name: str = "small"):
    global _model
    if _model is None:
        return preload_model(model_name)
    return _model


# ── Очистка текста ────────────────────────────────────────────────────────────
def clean_asr_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\wа-яёa-z0-9\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Запись аудио с возможностью досрочной остановки ──────────────────────────
def record_audio(filename: str = "input.wav", seconds: int = 7, fs: int = 16000):
    """
    Записывает аудио. Параллельно ждёт Enter:
      - Enter без текста  → досрочно останавливает запись, возвращает None
      - Enter с текстом   → останавливает запись, возвращает введённый текст
      - Таймаут           → останавливает запись, возвращает None (идёт транскрипция)
    Возвращает: None (использовать голос) или str (использовать текст напрямую)
    """
    import sounddevice as sd
    from scipy.io.wavfile import write

    # Ждём пока бот закончит говорить
    waited = False
    while is_bot_speaking():
        if not waited:
            print("[Voice] Жду окончания речи бота...")
            waited = True
        time.sleep(0.05)
    if waited:
        time.sleep(1.0)

    # Результат от потока ввода
    _input_result = [None]   # None = не нажали, str = ввели текст или пустую строку
    _stop_event = threading.Event()

    def _wait_for_enter():
        try:
            line = sys.stdin.readline()
            _input_result[0] = line.strip()
        except Exception:
            _input_result[0] = ""
        _stop_event.set()

    input_thread = threading.Thread(target=_wait_for_enter, daemon=True)

    print(f"[Voice] Говорите... ({seconds} сек)  |  Enter — остановить / ввести команду")
    audio_frames = []

    # Пишем чанками по 0.1 сек, проверяем стоп
    chunk = int(fs * 0.1)
    recorded_chunks = 0
    total_chunks = int(seconds * 10)  # секунды * 10 чанков в секунду

    input_thread.start()

    with sd.InputStream(samplerate=fs, channels=1, dtype="int16") as stream:
        while recorded_chunks < total_chunks:
            if _stop_event.is_set():
                break
            data, _ = stream.read(chunk)
            audio_frames.append(data)
            recorded_chunks += 1

    _stop_event.set()  # на случай если таймаут сработал раньше Enter

    # Сохраняем то что успели записать
    import numpy as np
    from scipy.io.wavfile import write as wav_write
    if audio_frames:
        audio = np.concatenate(audio_frames, axis=0)
        wav_write(filename, fs, audio)

    # Возвращаем текст если пользователь что-то напечатал
    typed = _input_result[0]
    if typed is not None and typed != "":
        # Напечатал текст — использовать его напрямую
        return typed
    # Нажал просто Enter или таймаут — использовать голос
    return None


# ── Транскрипция ──────────────────────────────────────────────────────────────
def speech_to_text(filename: str = "input.wav", model_name: str = "small") -> str:
    model = _get_model(model_name)
    result = model.transcribe(
        filename,
        language="ru",
        initial_prompt="Это разговор с чат-ботом на русском языке. Пользователь называет своё имя, спрашивает погоду, время, задаёт вопросы."
    )
    return result["text"]


# ── Главная функция ───────────────────────────────────────────────────────────
def listen(seconds: int = 7, model_name: str = "small") -> str:
    """
    Полный цикл:
      1. Ждёт окончания TTS
      2. Записывает микрофон (с возможностью прервать Enter)
      3. Если Enter с текстом — возвращает текст напрямую (команды и т.д.)
      4. Если Enter без текста или таймаут — транскрибирует голос
      5. Очищает и возвращает строку
    """
    typed = record_audio(seconds=seconds)

    if typed is not None:
        # Пользователь напечатал что-то — возвращаем как есть (без clean, чтобы команды работали)
        print(f"[Voice] Введено: «{typed}»")
        return typed

    # Голосовой ввод — транскрибируем
    raw = speech_to_text(model_name=model_name)
    clean = clean_asr_text(raw)
    if clean:
        print(f"[Voice] Распознано: «{clean}»")
    return clean