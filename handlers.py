from datetime import datetime
import string

class Handlers:
    def __init__(self):
        self.name = None
    
    def handle_greeting(self, match=None):
        if self.name:
            return f"Здравствуйте, {self.name}! Чем могу помочь?"
        return "Здравствуйте! Чем могу помочь?"
    
    def handle_farewell(self, match=None):
        return "До свидания!"
    
    def handle_set_name(self, match):
        self.name = match.group(1)
        return f"Приятно познакомиться, {self.name}!"
    
    def handle_weather(self, match):
        city = match.group(1).strip()
        return f"Погода в городе {city}: солнечно (демо-режим)."
    
    def handle_addition(self, match):
        try:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат: {a + b}"
        except ValueError:
            return "Не могу сложить эти числа."
    
    def handle_subtraction(self, match):
        try:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат: {a - b}"
        except ValueError:
            return "Не могу вычесть эти числа."

    def handle_how_are_you(self, match=None):
        return "У меня всё отлично, спасибо!"
    
    def handle_time(self, match=None):
        current_time = datetime.now().strftime("%H:%M")
        return f"Сейчас {current_time}"
    
    def handle_unknown(self, match=None):
        return "Не понимаю запрос."