import re

patterns = [
    (re.compile(r"привет|здравствуй|добрый день", re.IGNORECASE), "greeting"),
    
    (re.compile(r"пока|до свидания", re.IGNORECASE), "farewell"),
    
    (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), "set_name"),
    
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), "weather"),
    
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), "addition"),
    
    (re.compile(r"(\d+)\s*-\s*(\d+)"), "subtraction"),

    (re.compile(r"как (у тебя )?дела", re.IGNORECASE), "how_are_you"),
    
    (re.compile(r"(сколько|какое) (времени|время)", re.IGNORECASE), "time"),
]