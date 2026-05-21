

import re
import hashlib
import asyncio
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("tts_cache")
CACHE_DIR.mkdir(exist_ok=True)

_tts_enabled = True

VOICES = {
    "dmitry":   "ru-RU-DmitryNeural",
    "svetlana": "ru-RU-SvetlanaNeural",
    "dariya":   "ru-RU-DariyaNeural",
}

_current_voice = VOICES["dmitry"]


def set_voice(name: str):
    global _current_voice
    key = name.lower().strip()
    if key in VOICES:
        _current_voice = VOICES[key]
        return _current_voice
    return None


def get_voice() -> str:
    return _current_voice


def list_voices() -> str:
    lines = ["Доступные голоса:"]
    for name, full in VOICES.items():
        marker = " ◀ текущий" if full == _current_voice else ""
        lines.append(f"  /voice {name:<10} ({full}){marker}")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    # ссылки
    text = re.sub(r'https?://\S+', '', text)
    # markdown
    text = re.sub(r'[*_`~]', '', text)
    text = re.sub(u'[\U0001F300-\U0001F9FF]', '', text)
    text = re.sub(u'[\U00002600-\U000027BF]', '', text)
    text = re.sub(u'[\U0001FA00-\U0001FFFF]', '', text)
    # скобочные смайлики :) :D ;( и т.д.
    text = re.sub(r'[:;=]-?[\)\(DdPpOo\[\]/\\]', '', text)
    # двойные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _get_cache_path(text: str, voice: str) -> Path:
    key = hashlib.md5(f"{voice}:{text}".encode('utf-8')).hexdigest()
    return CACHE_DIR / f"{key}.mp3"


async def _synthesize_async(text: str, voice: str, output_path: Path) -> bool:
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except ImportError:
        logger.error("edge-tts не установлен. Запустите: pip install edge-tts")
        return False
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}")
        return False


def _synthesize(text: str, voice: str):
    cache_path = _get_cache_path(text, voice)
    if cache_path.exists():
        return cache_path
    success = asyncio.run(_synthesize_async(text, voice, cache_path))
    return cache_path if success else None


def _play(file_path: Path) -> None:
    import platform
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            # Используем Windows MCI напрямую — играет mp3 без лишних библиотек
            winmm = ctypes.windll.winmm
            path = str(file_path.resolve())
            winmm.mciSendStringW(f'open "{path}" type mpegvideo alias media', None, 0, None)
            winmm.mciSendStringW('play media wait', None, 0, None)
            winmm.mciSendStringW('close media', None, 0, None)
        elif system == "Darwin":
            import subprocess
            subprocess.run(["afplay", str(file_path)], check=True)
        else:
            import subprocess
            subprocess.run(["mpg123", "-q", str(file_path)], check=True)
    except Exception as e:
        logger.error(f"Ошибка воспроизведения: {e}")


def speak(text: str) -> None:
    if not _tts_enabled or not text or not text.strip():
        return
    clean = normalize_text(text)
    if not clean:
        return
    voice = _current_voice

    # Выставляем флаг ДО запуска треда — voice.py сразу заблокирует запись
    try:
        import voice as _voice_mod
        _voice_mod.set_speaking(True)
    except ImportError:
        _voice_mod = None

    def _worker():
        try:
            path = _synthesize(clean, voice)
            if path:
                _play(path)
        finally:
            if _voice_mod is not None:
                _voice_mod.set_speaking(False)

    threading.Thread(target=_worker, daemon=True).start()


def toggle_tts() -> bool:
    global _tts_enabled
    _tts_enabled = not _tts_enabled
    return _tts_enabled


def is_enabled() -> bool:
    return _tts_enabled


def clear_cache() -> int:
    count = 0
    for f in CACHE_DIR.glob("*.mp3"):
        f.unlink()
        count += 1
    return count