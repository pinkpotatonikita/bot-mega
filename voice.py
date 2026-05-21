import re
import time
import threading
import logging

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


# ── Транскрипция ──────────────────────────────────────────────────────────────
def speech_to_text(filename: str = "input.wav", model_name: str = "small") -> str:
    model = _get_model(model_name)
    result = model.transcribe(
        filename,
        language="ru",
        initial_prompt="Это разговор с чат-ботом на русском языке. Пользователь называет своё имя, спрашивает погоду, время, задаёт вопросы.",
    )
    return result["text"]


# ── PTT: запись пока зажат ПРОБЕЛ ────────────────────────────────────────────
def listen_ptt(model_name: str = "small") -> str:
    """
    PTT-цикл (Push-To-Talk):
      1. Ждёт окончания TTS бота
      2. Выводит «Зажмите ПРОБЕЛ чтобы говорить»
      3. При нажатии ПРОБЕЛА — начинает запись, выводит «🔴 Говорите...»
      4. При отпускании ПРОБЕЛА — останавливает запись
      5. Транскрибирует и возвращает текст

    Требует: pip install pynput
    """
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write as wav_write

    try:
        from pynput import keyboard as kb
    except ImportError:
        logger.error("pynput не установлен: pip install pynput")
        raise

    # Ждём пока бот закончит говорить
    if is_bot_speaking():
        print("[Voice] Жду окончания речи бота...")
        while is_bot_speaking():
            time.sleep(0.05)
        time.sleep(0.3)

    print("Зажмите ПРОБЕЛ чтобы говорить...")

    fs = 16000
    filename = "input.wav"

    space_down = threading.Event()
    space_up   = threading.Event()
    esc_pressed = threading.Event()

    def on_press(key):
        if key == kb.Key.space and not space_down.is_set():
            space_down.set()
        elif key == kb.Key.esc:
            esc_pressed.set()
            space_down.set()   # разблокируем ожидание

    def on_release(key):
        if key == kb.Key.space:
            space_up.set()
            return False  # останавливаем listener
        elif key == kb.Key.esc:
            return False

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Ждём нажатия пробела (или Esc для выхода)
    print("Зажмите ПРОБЕЛ чтобы говорить  |  ESC — выйти из голосового режима")
    space_down.wait()

    if esc_pressed.is_set():
        listener.stop()
        return "\x1b"  # сигнал выхода

    print("🔴 Говорите...")

    # Запись в фоне пока пробел зажат
    audio_frames = []
    stop_rec = threading.Event()

    def _record():
        chunk = int(fs * 0.05)
        with sd.InputStream(samplerate=fs, channels=1, dtype="int16") as stream:
            while not stop_rec.is_set():
                data, _ = stream.read(chunk)
                audio_frames.append(data)

    rec_thread = threading.Thread(target=_record, daemon=True)
    rec_thread.start()

    space_up.wait()          # ждём отпускания
    stop_rec.set()
    rec_thread.join(timeout=2.0)
    listener.stop()

    if not audio_frames:
        return ""

    audio = np.concatenate(audio_frames, axis=0)
    wav_write(filename, fs, audio)

    raw   = speech_to_text(filename, model_name)
    clean = clean_asr_text(raw)
    return clean


# listen() оставлен для обратной совместимости, но просто вызывает PTT
def listen(seconds: int = 7, model_name: str = "small") -> str:
    return listen_ptt(model_name=model_name)