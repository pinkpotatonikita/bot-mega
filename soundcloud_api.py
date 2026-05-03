import re
import json
import random
import urllib.request
from datetime import datetime, timedelta

# ── Маппинг настроений → поисковые запросы SoundCloud ────────────────────────

MOOD_QUERIES = {
    "грустный":     ["sad", "melancholy", "heartbreak"],
    "грустно":      ["sad", "melancholy"],
    "грусть":       ["sad rainy", "melancholy"],
    "печаль":       ["sad", "melancholy"],
    "тоска":        ["sad", "lonely"],
    "плохо":        ["sad", "melancholy"],

    "весёлый":      ["happy", "fun upbeat"],
    "веселье":      ["happy feel good"],
    "радостный":    ["happy", "uplifting"],
    "радость":      ["uplifting happy"],
    "хорошо":       ["feel good", "happy"],
    "счастье":      ["happy joyful"],

    "энергичный":   ["energetic", "pump up"],
    "энергия":      ["energetic workout"],
    "тренировка":   ["workout", "gym motivation"],
    "спорт":        ["workout running"],
    "бодрый":       ["energetic morning"],

    "спокойный":    ["chill", "relaxing"],
    "спокойно":     ["chill ambient"],
    "расслабление": ["relaxing lounge"],
    "отдых":        ["chill lounge"],
    "медитация":    ["meditation ambient"],

    "романтика":    ["romantic", "love songs"],
    "романтичный":  ["romantic"],
    "любовь":       ["love songs"],

    "ночь":         ["late night", "night vibes"],
    "сон":          ["sleep ambient", "calm sleep"],
    "ночной":       ["night vibes", "dark ambient"],

    "утро":         ["morning acoustic", "wake up"],
    "утренний":     ["morning chill"],

    "работа":       ["focus study", "concentration"],
    "учёба":        ["study music", "focus"],
    "фокус":        ["focus instrumental"],
    "концентрация": ["concentration study"],
}

DEFAULT_QUERIES = ["chill", "indie", "alternative"]


class SoundCloudAPI:
    _API_BASE = "https://api-v2.soundcloud.com"
    _SC_URL   = "https://soundcloud.com"

    def __init__(self, client_id: str = ""):
        self._client_id  = client_id
        self._cache: dict = {}
        self._cache_ttl  = timedelta(minutes=30)

    # ── Получение client_id ───────────────────────────────────────────────────

    def _fetch_client_id(self) -> str:
        """Автоматически извлекает актуальный client_id из JS-бандлов SoundCloud."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        # 1. Загружаем главную страницу
        req = urllib.request.Request(self._SC_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            html = r.read().decode("utf-8", errors="ignore")

        # 2. Находим ссылки на JS-бандлы
        scripts = re.findall(
            r'src="(https://a-v2\.sndcdn\.com/assets/[^"]+\.js)"', html
        )
        if not scripts:
            raise RuntimeError("JS-бандлы SoundCloud не найдены")

        # 3. Ищем client_id в последних 5 бандлах
        for url in reversed(scripts[-5:]):
            try:
                req2 = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req2, timeout=8) as r2:
                    js = r2.read().decode("utf-8", errors="ignore")
                matches = re.findall(r'client_id:"([a-zA-Z0-9]{32})"', js)
                if matches:
                    return matches[0]
            except Exception:
                continue

        raise RuntimeError("client_id не найден в JS-бандлах SoundCloud")

    def _get_client_id(self) -> str:
        """Возвращает client_id — из кеша, настроек или автоматически."""
        if self._client_id:
            return self._client_id
        # Пробуем получить автоматически и кешируем
        cid = self._fetch_client_id()
        self._client_id = cid
        return cid

    # ── Поиск треков ──────────────────────────────────────────────────────────

    def _search(self, query: str, limit: int = 20) -> list[dict]:
        """Ищет треки на SoundCloud по запросу."""
        cache_key = f"search:{query}:{limit}"
        if cache_key in self._cache:
            data, ts = self._cache[cache_key]
            if datetime.now() - ts < self._cache_ttl:
                return data

        client_id = self._get_client_id()
        import urllib.parse
        params = urllib.parse.urlencode({
            "q":         query,
            "client_id": client_id,
            "limit":     limit,
            "offset":    0,
            "linked_partitioning": 1,
        })
        url = f"{self._API_BASE}/search/tracks?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())

        tracks = [
            {
                "artist": t["user"]["username"],
                "title":  t["title"],
                "url":    t["permalink_url"],
                "genre":  t.get("genre", ""),
            }
            for t in data.get("collection", [])
            if isinstance(t, dict) and t.get("permalink_url")
        ]
        self._cache[cache_key] = (tracks, datetime.now())
        return tracks

    # ── Определение настроения ────────────────────────────────────────────────

    def detect_mood(self, text: str) -> tuple[str, list[str]]:
        text_lower = text.lower()
        for mood, queries in MOOD_QUERIES.items():
            if mood in text_lower:
                return mood, queries
        return "нейтральное", DEFAULT_QUERIES

    # ── Основной метод ────────────────────────────────────────────────────────

    def get_playlist(self, text: str, count: int = 10) -> dict:
        """Определяет настроение и собирает плейлист из SoundCloud."""
        mood, queries = self.detect_mood(text)

        all_tracks: list[dict] = []
        for query in queries:
            try:
                tracks = self._search(query, limit=30)
                all_tracks.extend(tracks)
            except Exception:
                continue

        if not all_tracks:
            return {"error": "Не удалось получить треки с SoundCloud"}

        # Убираем дубликаты по URL
        seen = set()
        unique = []
        for t in all_tracks:
            if t["url"] not in seen:
                seen.add(t["url"])
                unique.append(t)

        random.shuffle(unique)
        return {
            "mood":   mood,
            "tracks": unique[:count],
        }

    # ── Форматирование ────────────────────────────────────────────────────────

    def format_playlist(self, data: dict) -> str:
        if "error" in data:
            return f"❌ {data['error']}"

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
        mood   = data["mood"]
        tracks = data["tracks"]
        emoji  = MOOD_EMOJI.get(mood, "🎵")

        lines = [f"{emoji} Плейлист по настроению «{mood}» — SoundCloud:\n"]
        for i, t in enumerate(tracks, 1):
            lines.append(f"{i:2}. {t['artist']} — {t['title']}")
            lines.append(f"    🔗 {t['url']}")
        return "\n".join(lines)


# ── Синглтон ──────────────────────────────────────────────────────────────────
# client_id подхватится автоматически из JS SoundCloud при первом запросе.
soundcloud_api = SoundCloudAPI()