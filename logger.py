import os
from datetime import datetime

def log_message(user_message, bot_response):
    # current_dir = os.getcwd()
    # file_path = os.path.join(current_dir, "chat_log.txt")
    
    # print(f" Лог сохраняется в: {file_path}") 
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] USER: {user_message}\n")
        f.write(f"[{datetime.now()}] BOT: {bot_response}\n")
        f.write("-" * 50 + "\n")