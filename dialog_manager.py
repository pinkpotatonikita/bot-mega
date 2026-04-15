from enum import Enum


class DialogState(Enum):
    START        = "start"
    WAIT_NAME    = "wait_name"   # ждём имя пользователя
    WAIT_CITY    = "wait_city"   # ждём название города
    WAIT_DATE    = "wait_date"   # ждём дату (доп. задание)


class DialogManager:
    """
    Конечный автомат (FSM) для управления состоянием диалога.

    Переходы:
        START  ──► WAIT_NAME   (приветствие без имени)
        START  ──► WAIT_CITY   (запрос погоды без города)
        START  ──► WAIT_DATE   (после получения города — спрашиваем дату)
        WAIT_* ──► START       (после обработки — сброс)
    """

    def __init__(self):
        self._states: dict[int, DialogState] = {}
        self._data:   dict[int, dict]        = {}

    # ── Состояние ─────────────────────────────────────────────────────────────

    def get_state(self, user_id: int) -> DialogState:
        return self._states.get(user_id, DialogState.START)

    def set_state(self, user_id: int, state: DialogState) -> None:
        self._states[user_id] = state

    # ── Данные сессии ─────────────────────────────────────────────────────────

    def get_data(self, user_id: int) -> dict:
        if user_id not in self._data:
            self._data[user_id] = {}
        return self._data[user_id]

    def set_data(self, user_id: int, key: str, value) -> None:
        self.get_data(user_id)[key] = value

    # ── Сброс ─────────────────────────────────────────────────────────────────

    def reset(self, user_id: int) -> None:
        self._states[user_id] = DialogState.START
        self._data[user_id]   = {}


# Глобальный синглтон — импортируется всеми модулями
dialog_manager = DialogManager()