import os
from datetime import datetime
from database import Database

# Глобальный экземпляр базы данных
_db = Database()

def log_message(user_message, bot_response, user_name=None):
    """Сохраняет лог сообщения в файл и базу данных"""
    
    # Сохраняем в текстовый файл (для обратной совместимости)
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] USER: {user_message}\n")
        f.write(f"[{datetime.now()}] BOT: {bot_response}\n")
        f.write("-" * 50 + "\n")
    
    # Сохраняем в базу данных
    _db.save_log(user_message, bot_response, user_name)

def get_db():
    """Возвращает экземпляр базы данных"""
    return _db