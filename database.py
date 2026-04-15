import sqlite3
from datetime import datetime
import os

class Database:
    def __init__(self, db_name="chatbot.db"):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        """Создает и возвращает соединение с БД"""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Инициализация таблиц в базе данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    interactions_count INTEGER DEFAULT 1
                )
            ''')
            
            # Таблица логов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
    
    def get_or_create_user(self, name):
        """Получает пользователя по имени или создает нового"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Пытаемся найти пользователя
            cursor.execute('SELECT * FROM users WHERE name = ?', (name,))
            user = cursor.fetchone()
            
            if user:
                # Обновляем существующего пользователя
                cursor.execute('''
                    UPDATE users 
                    SET last_interaction = CURRENT_TIMESTAMP,
                        interactions_count = interactions_count + 1
                    WHERE id = ?
                ''', (user['id'],))
                return user['id']
            else:
                # Создаем нового пользователя
                cursor.execute('''
                    INSERT INTO users (name, first_seen, last_interaction)
                    VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (name,))
                return cursor.lastrowid
    
    def save_log(self, user_message, bot_response, user_name=None):
        """Сохраняет лог сообщения в базу данных"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            user_id = None
            if user_name:
                user_id = self.get_or_create_user(user_name)
            
            cursor.execute('''
                INSERT INTO chat_logs (user_id, user_message, bot_response)
                VALUES (?, ?, ?)
            ''', (user_id, user_message, bot_response))
            
            conn.commit()
    
    def get_user_stats(self, user_name=None):
        """Получает статистику по пользователю или всем пользователям"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if user_name:
                cursor.execute('''
                    SELECT u.*, COUNT(l.id) as total_messages
                    FROM users u
                    LEFT JOIN chat_logs l ON u.id = l.user_id
                    WHERE u.name = ?
                    GROUP BY u.id
                ''', (user_name,))
            else:
                cursor.execute('''
                    SELECT u.*, COUNT(l.id) as total_messages
                    FROM users u
                    LEFT JOIN chat_logs l ON u.id = l.user_id
                    GROUP BY u.id
                    ORDER BY u.last_interaction DESC
                ''')
            
            return cursor.fetchall()
    
    def get_recent_logs(self, limit=10):
        """Получает последние логи"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT l.*, u.name as user_name
                FROM chat_logs l
                LEFT JOIN users u ON l.user_id = u.id
                ORDER BY l.timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()
    
    def close(self):
        """Закрывает соединение с БД"""
        pass  # connection autoclosed by context manager