import threading
from difflib import SequenceMatcher

class ResponseWriter:
    def __init__(self):
        self.previous_message = ""
        self._lock = threading.Lock()

    def normalize_text(self, text: str) -> str:
        text = text.replace('\\"', '\\\" ')
        text = text.replace(':\"', ':\" ')
        text = text.replace('""""', '###')
        text = text.replace('"""', '```')
        text = text.replace('":', '" :')
        return text

    def get_delta_content(self, current_message: str, threshold: float = 0.6) -> str:
        with self._lock:
            if not self.previous_message:
                self.previous_message = current_message
                return current_message
                
            sm = SequenceMatcher(None, self.previous_message, current_message)
            ratio = sm.ratio()
            
            if 1 - ratio > threshold:
                self.previous_message = current_message
                return current_message
                
            delta = ""
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ('replace', 'insert'):
                    delta += current_message[j1:j2]
                    
            self.previous_message = current_message
            return delta
