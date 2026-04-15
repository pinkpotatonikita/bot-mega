import re
import random
from datetime import datetime, timedelta

import spacy
from spacy.matcher import Matcher

from weather_api import get_weather, get_weather_forecast
from database import init_db, save_user, log_weather_query
from dialog_manager import dialog_manager, DialogState  

nlp = spacy.load("ru_core_news_sm")

matcher = Matcher(nlp.vocab)
matcher.add("WEATHER_QUERY", [
    [{"LEMMA": "погода"}, {"LOWER": {"IN": ["в", "во"]}}, {"POS": "PROPN"}],
    [{"LEMMA": "погода"}, {"LOWER": {"IN": ["в", "во"]}}, {"IS_TITLE": True}],
    [{"LEMMA": "погода"}, {"OP": "?"}, {"LOWER": {"IN": ["в", "во"]}}, {"POS": "PROPN"}],
])

DATE_KEYWORDS = {
    "сегодня":   0,
    "завтра":    1,
    "послезавтра": 2,
    "через два дня": 2,
    "через 2 дня":   2,
}

WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "среду": 2, "среда": 2,
    "четверг": 3, "пятницу": 4, "пятница": 4,
    "субботу": 5, "суббота": 5, "воскресенье": 6, "воскресенье": 6,
}


def extract_date_offset(text: str) -> tuple[int, str]:
    text_lower = text.lower()

    for keyword, offset in DATE_KEYWORDS.items():
        if keyword in text_lower:
            labels = {0: "сегодня", 1: "завтра", 2: "послезавтра"}
            return offset, labels.get(offset, f"через {offset} дня")

    today_wd = datetime.now().weekday()
    for word, target_wd in WEEKDAYS.items():
        if word in text_lower:
            offset = (target_wd - today_wd) % 7
            if offset == 0:
                offset = 7
            return offset, word

    return 0, "сегодня"


def extract_city(text: str) -> str | None:
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            lemma = ent.root.lemma_.capitalize()
            print(f"[DEBUG] NER: {ent.text!r} -> {lemma!r}")
            return lemma

    m = re.search(r"\bв[о]?\s+([А-ЯЁ][а-яёА-ЯЁ\-]+)", text)
    if m:
        word = m.group(1)
        word_doc = nlp(word)
        lemma = word_doc[0].lemma_.capitalize() if word_doc else word
        print(f"[DEBUG] Regex: {word!r} -> {lemma!r}")
        return lemma

    print(f"[DEBUG] Город не найден в: {text!r}")
    return None


def is_weather_query(text: str) -> bool:
    doc = nlp(text)
    if matcher(doc):
        return True
    return "погода" in text.lower()


class ChatBot:
    def __init__(self):
        self.name = None
        self.current_user_id = None
        self.waiting_for_name = False
        self.patterns = []
        init_db()
        self.register_patterns()

    def register_patterns(self):
        self.patterns.append(
            (re.compile(r"^(привет|здравствуй|добрый день|здравствуйте|прив)$", re.IGNORECASE), self.greet)
        )
        self.patterns.append(
            (re.compile(r"^(пока|до свидания|выход)$", re.IGNORECASE), self.farewell)
        )
        self.patterns.append(
            (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), self.set_name)
        )
        self.patterns.append(
            (re.compile(r"как (у тебя )?дела", re.IGNORECASE), self.how_are_you)
        )
        self.patterns.append(
            (re.compile(r"(сколько|какое) (сейчас )?время", re.IGNORECASE), self.what_time)
        )
        self.patterns.append(
            (re.compile(r"(\d+)\s*\+\s*(\d+)"), self.handle_addition)
        )

    def greet(self, match):
        if self.name:
            return f"Здравствуйте, {self.name}! Чем могу помочь?"
        self.waiting_for_name = True
        return "Здравствуйте! Чем могу помочь? Как вас зовут?"

    def farewell(self, match):
        if self.name:
            return f"До свидания, {self.name}! Было приятно пообщаться."
        return "До свидания!"

    def set_name(self, match):
        self.name = match.group(1).capitalize()
        self.current_user_id = save_user(self.name)
        if self.current_user_id:
            dialog_manager.reset(self.current_user_id)
        return f"Приятно познакомиться, {self.name}!"

    def handle_weather_with_state(self, message: str) -> str:
        if not self.current_user_id:
            return "Пожалуйста, представьтесь сначала (например: 'меня зовут Иван')"
        
        user_id = self.current_user_id
        state = dialog_manager.get_state(user_id)
        data = dialog_manager.get_data(user_id)
        
        print(f"[DEBUG] State: {state.value}, Message: {message}")
        
        if state == DialogState.START:
            if is_weather_query(message):
                city = extract_city(message)
                
                if city:
                   
                    offset, day_label = extract_date_offset(message)
                    if user_id:
                        log_weather_query(user_id, city)
                    
                    if offset == 0:
                        return get_weather(city)
                    else:
                        return get_weather_forecast(city, offset, day_label)
                else:
                    
                    dialog_manager.set_state(user_id, DialogState.WAIT_CITY)
                    return "В каком городе вас интересует погода?"
            return None  
        
        # Состояние WAIT_CITY - ожидаем город
        elif state == DialogState.WAIT_CITY:
            city = extract_city(f"в {message}")  
            if not city:
                city = message.strip().capitalize() 
            
          
            data['city'] = city
            offset, day_label = extract_date_offset(message)
            data['offset'] = offset
            data['day_label'] = day_label
            
            dialog_manager.set_state(user_id, DialogState.WAIT_DATE)
            return f"На какую дату вас интересует погода в городе {city}?"
            
       # # Базовое задание - сразу показываем погоду
        #if user_id:
            #log_weather_query(user_id, city)
            
            #result = get_weather(city) if offset == 0 else get_weather_forecast(city, offset, day_label)
           # dialog_manager.reset(user_id)
           # return result
        
        # Состояние WAIT_DATE (для доп. задания)
        elif state == DialogState.WAIT_DATE:
            city = data.get('city')
            if not city:
                dialog_manager.reset(user_id)
                return "Произошла ошибка. Начните заново."
            
            offset, day_label = extract_date_offset(message)
            
            if user_id:
                log_weather_query(user_id, city)
            
            result = get_weather(city) if offset == 0 else get_weather_forecast(city, offset, day_label)
            dialog_manager.reset(user_id)
            return result
        
        return None

    def handle_addition(self, match):
        try:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат сложения: {a} + {b} = {a + b}"
        except Exception:
            return "Не удалось выполнить сложение"

    def how_are_you(self, match):
        return random.choice([
            "Всё отлично, спасибо!",
            "Хорошо, а у вас?",
            "Прекрасно! Готов помочь.",
        ])

    def what_time(self, match):
        return f"Сейчас {datetime.now().strftime('%H:%M')}"

    def process(self, message: str) -> str:
        message_clean = message.strip()

        
        if self.waiting_for_name and re.match(r'^[а-яА-ЯёЁa-zA-Z]+$', message_clean):
            self.waiting_for_name = False
            self.name = message_clean.capitalize()
            self.current_user_id = save_user(self.name)
            if self.current_user_id:
                dialog_manager.reset(self.current_user_id)
            return f"Приятно познакомиться, {self.name}!"

     
        if self.current_user_id:
            state_response = self.handle_weather_with_state(message_clean)
            if state_response:
                return state_response

      
        for pattern, handler in self.patterns:
            m = pattern.search(message)
            if m:
                return handler(m)

      
        return "Я не понимаю этот запрос. Попробуйте спросить о погоде, времени или просто поздороваться."


bot = ChatBot()