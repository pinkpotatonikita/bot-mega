import requests
from datetime import datetime, timedelta

# ── Маппинг настроений → теги Last.fm ────────────────────────────────────────

MOOD_TAGS = {
    # грусть
    "грустный":    ["sad", "melancholy", "heartbreak"],
    "грустно":     ["sad", "melancholy", "heartbreak"],
    "грусть":      ["sad", "melancholy", "heartbreak"],
    "печаль":      ["sad", "melancholy", "rainy day"],
    "тоска":       ["sad", "melancholy", "lonely"],
    "плохо":       ["sad", "melancholy"],

    # радость / веселье
    "весёлый":     ["happy", "feel good", "fun"],
    "веселье":     ["happy", "feel good", "fun"],
    "радостный":   ["happy", "uplifting", "feel good"],
    "радость":     ["happy", "uplifting"],
    "хорошо":      ["happy", "feel good", "upbeat"],
    "счастье":     ["happy", "joyful", "uplifting"],

    # энергия / спорт
    "энергичный":  ["energetic", "workout", "pump up"],
    "энергия":     ["energetic", "pump up"],
    "тренировка":  ["workout", "energetic", "pump up"],
    "спорт":       ["workout", "energetic", "running"],
    "бодрый":      ["energetic", "upbeat", "morning"],

    # расслабление
    "спокойный":   ["chill", "relaxing", "calm"],
    "спокойно":    ["chill", "relaxing", "ambient"],
    "расслабление":["chill", "relaxing", "lounge"],
    "отдых":       ["chill", "relaxing", "lounge"],
    "медитация":   ["meditation", "ambient", "relaxing"],

    # романтика
    "романтика":   ["romantic", "love", "romance"],
    "романтичный": ["romantic", "love"],
    "любовь":      ["love", "romantic"],

    # ночь / сон
    "ночь":        ["night", "late night", "sleep"],
    "сон":         ["sleep", "ambient", "calm"],
    "ночной":      ["night", "late night", "dark"],

    # утро
    "утро":        ["morning", "wake up", "acoustic"],
    "утренний":    ["morning", "acoustic", "feel good"],

    # концентрация / работа
    "работа":      ["focus", "study", "concentration"],
    "учёба":       ["study", "focus", "instrumental"],
    "фокус":       ["focus", "concentration", "study"],
    "концентрация":["concentration", "focus", "study"],

    # грусть по умолчанию / нейтральное
    "нейтральный": ["indie", "alternative", "pop"],
}

# Дефолтные теги если настроение не распознано
DEFAULT_TAGS = ["pop", "indie", "alternative"]


class LastFMApi:
    BASE_URL = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: dict = {}
        self._cache_ttl = timedelta(minutes=30)

    # ── Определение настроения из текста ─────────────────────────────────────

    def detect_mood(self, text: str) -> tuple[str, list[str]]:
        """
        Находит настроение в тексте пользователя.
        Возвращает (название настроения, список тегов для Last.fm).
        """
        text_lower = text.lower()
        for mood, tags in MOOD_TAGS.items():
            if mood in text_lower:
                return mood, tags
        return "нейтральное", DEFAULT_TAGS

    # ── Запросы к Last.fm API ─────────────────────────────────────────────────

    def get_tracks_by_tag(self, tag: str, limit: int = 50) -> list[dict]:
        """Получает топ треков по тегу настроения."""
        cache_key = f"tag:{tag}:{limit}"
        if cache_key in self._cache:
            data, ts = self._cache[cache_key]
            if datetime.now() - ts < self._cache_ttl:
                return data

        try:
            resp = requests.get(self.BASE_URL, params={
                "method":  "tag.gettoptracks",
                "tag":     tag,
                "api_key": self.api_key,
                "format":  "json",
                "limit":   limit,
            }, timeout=6)
            resp.raise_for_status()
            tracks = resp.json().get("tracks", {}).get("track", [])
            result = [
                {
                    "artist": t["artist"]["name"],
                    "title":  t["name"],
                    "url":    t.get("url", ""),
                }
                for t in tracks if isinstance(t, dict)
            ]
            self._cache[cache_key] = (result, datetime.now())
            return result
        except Exception:
            return []

    def get_playlist(self, text: str, count: int = 10) -> dict:
        """
        Основной метод: определяет настроение → собирает плейлист.
        Возвращает dict с настроением и списком треков.
        """
        import random
        mood, tags = self.detect_mood(text)

        all_tracks: list[dict] = []
        for tag in tags:
            tracks = self.get_tracks_by_tag(tag, limit=50)
            all_tracks.extend(tracks)

        # Убираем дубликаты по "artist - title"
        seen = set()
        unique: list[dict] = []
        for t in all_tracks:
            key = f"{t['artist'].lower()}|{t['title'].lower()}"
            if key not in seen:
                seen.add(key)
                unique.append(t)

        if not unique:
            return {"error": "Не удалось получить треки с Last.fm"}

        random.shuffle(unique)
        return {
            "mood":   mood,
            "tags":   tags,
            "tracks": unique[:count],
        }

    # ── Форматирование ────────────────────────────────────────────────────────

    def format_playlist(self, data: dict) -> str:
        if "error" in data:
            return f"❌ {data['error']}"

        mood  = data["mood"]
        tracks = data["tracks"]

        MOOD_EMOJI = {
            "грустный": "😢", "грустно": "😢", "грусть": "😢",
            "печаль": "😔", "тоска": "😔", "плохо": "😔",
            "весёлый": "😄", "веселье": "😄", "радостный": "😄",
            "радость": "😄", "хорошо": "😊", "счастье": "😄",
            "энергичный": "⚡", "энергия": "⚡", "тренировка": "💪",
            "спорт": "🏃", "бодрый": "⚡",
            "спокойный": "😌", "спокойно": "😌", "расслабление": "🛋️",
            "отдых": "🛋️", "медитация": "🧘",
            "романтика": "❤️", "романтичный": "❤️", "любовь": "❤️",
            "ночь": "🌙", "сон": "🌙", "ночной": "🌙",
            "утро": "🌅", "утренний": "🌅",
            "работа": "💼", "учёба": "📚", "фокус": "🎯",
            "концентрация": "🎯",
        }
        emoji = MOOD_EMOJI.get(mood, "🎵")

        lines = [f"{emoji} Плейлист по настроению «{mood}»:\n"]
        for i, t in enumerate(tracks, 1):
            lines.append(f"{i:2}. {t['artist']} — {t['title']}")
        lines.append(
            f"\n🔍 Найти в Яндекс Музыке: https://music.yandex.ru/search?text="
            + "+".join(tracks[0]["artist"].split()) if tracks else ""
        )
        return "\n".join(lines)


# ── Синглтон ──────────────────────────────────────────────────────────────────
#  бесплатный ключ на https://www.last.fm/api/account/create
LASTFM_API_KEY = "271dc22d44074541e45a415c748517a8"
lastfm_api = LastFMApi(LASTFM_API_KEY)