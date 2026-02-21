import re
import string
from patterns import patterns
from handlers import Handlers
from logger import log_message

class ChatBot:
    def __init__(self):
        self.handlers = Handlers()
        self.patterns = patterns
        
        #ключ - метод обработчик
        self.handler_map = {
            "greeting": self.handlers.handle_greeting,
            "farewell": self.handlers.handle_farewell,
            "set_name": self.handlers.handle_set_name,
            "weather": self.handlers.handle_weather,
            "addition": self.handlers.handle_addition,
            "subtraction": self.handlers.handle_subtraction,
            "how_are_you": self.handlers.handle_how_are_you,
            "time": self.handlers.handle_time,
        }
    
    def preprocess_message(self, message):
        message = message.lower().strip()
        
        message = message.translate(str.maketrans('', '', string.punctuation))
        
        return message
    
    def process(self, message):

        original_message = message
        processed_message = self.preprocess_message(message)
        
        for pattern, handler_key in self.patterns:
            match = pattern.search(processed_message) or pattern.search(original_message)
            if match:
                handler = self.handler_map.get(handler_key, self.handlers.handle_unknown)
                response = handler(match)
                log_message(original_message, response)
                return response
            
        response = self.handlers.handle_unknown()
        log_message(original_message, response)

        return response

def main():
    bot = ChatBot()
    
    print("Чат-бот запущен. Введите 'пока' для выхода.")
    print("-" * 40)
    
    while True:
        user_input = input("Вы: ")
        
        if user_input.lower().strip() in ['пока', 'до свидания', 'exit', 'quit']:
            response = bot.handlers.handle_farewell()
            print(f"Бот: {response}")
            log_message(user_input, response)
            break
        
        response = bot.process(user_input)
        
        print(f"Бот: {response}")
        
        log_message(user_input, response)

if __name__ == "__main__":
    main()