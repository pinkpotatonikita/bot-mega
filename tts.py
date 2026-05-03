"""
Модуль озвучки для чат-бота Димсн.
Использует gTTS (Google Text-to-Speech) для русского языка.
Поддерживает асинхронное воспроизведение и кэширование.
"""

import os
import re
import hashlib
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Папка для кэша аудио файлов
CACHE_DIR = Path("tts_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Флаг — включена ли озвучка (можно выключить командой /tts)
_tts_enabled = True

# Словарь замен для нормализации текста перед озвучкой
_REPLACEMENTS = [
    (r'\d+', lambda m: _digits_to_words(m.group())),  # числа → слова (упрощённо)
    (r'https?://\S+', ''),          # убираем ссылки
    (r'[*_`~]', ''),                # убираем markdown символы
    (r'\s+', ' '),                  # двойные пробелы
]


def _digits_to_words(num_str: str) -> str:
    """Простая замена — TTS и так нормально читает цифры, оставляем как есть."""
    return num_str


def normalize_text(text: str) -> str:
    """Убирает из текста лишнее перед озвучкой."""
    text = re.sub(r'https?://\S+', '', text)   # ссылки
    text = re.sub(r'[*_`~]', '', text)          # markdown
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _get_cache_path(text: str) -> Path:
    """Возвращает путь к кэш-файлу для данного текста."""
    key = hashlib.md5(text.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"{key}.mp3"


def _synthesize(text: str) -> Path | None:
    """Синтезирует речь и сохраняет в кэш. Возвращает путь к файлу."""
    cache_path = _get_cache_path(text)
    if cache_path.exists():
        return cache_path  # уже есть в кэше

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='ru', slow=False)
        tts.save(str(cache_path))
        return cache_path
    except ImportError:
        logger.error("gTTS не установлен. Запустите: pip install gtts")
        return None
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}")
        return None


def _play(file_path: Path) -> None:
    """Воспроизводит аудио файл через pygame."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(str(file_path))
        pygame.mixer.music.play()
        # Ждём окончания воспроизведения
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except ImportError:
        logger.error("pygame не установлен. Запустите: pip install pygame")
    except Exception as e:
        logger.error(f"Ошибка воспроизведения: {e}")


def speak(text: str) -> None:
    """
    Озвучивает текст асинхронно (не блокирует бота).
    Если озвучка выключена — ничего не делает.
    """
    global _tts_enabled
    if not _tts_enabled or not text or not text.strip():
        return

    clean = normalize_text(text)
    if not clean:
        return

    def _worker():
        path = _synthesize(clean)
        if path:
            _play(path)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def toggle_tts() -> bool:
    """Включает/выключает озвучку. Возвращает новое состояние."""
    global _tts_enabled
    _tts_enabled = not _tts_enabled
    return _tts_enabled


def is_enabled() -> bool:
    return _tts_enabled


def clear_cache() -> int:
    """Удаляет все кэшированные аудио файлы. Возвращает количество удалённых."""
    count = 0
    for f in CACHE_DIR.glob("*.mp3"):
        f.unlink()
        count += 1
    return count