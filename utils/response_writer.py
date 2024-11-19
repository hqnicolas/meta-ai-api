import threading
from difflib import SequenceMatcher
import re

class ResponseWriter:
    def __init__(self):
        self.previous_message = ""
        self._lock = threading.Lock()

    def normalize_text(self, text: str) -> str:
        text = text.replace('\\"', '\\\" ')
        text = text.replace(':\"', ':\" ')
        text = text.replace('""""', '###')
        text = text.replace('":', '" :')
        text = text.replace('**', '###')
        text = text.replace('/n# ', '### ')
        text = re.sub(r'(\w+)(?=\s\w)', r'\1` ', text)
        text = re.sub(r'/n`', r'/n````', text)
        text = re.sub(r'(`^`[]+)(?=\s.,!?[])', r'\1`', text)
        text = re.sub(r'(\w+)\s(?=[^`])', r'\1` ', text)
        text = re.sub(r'(\w+)\s`(\w+)(?=\s)', r'\1 `\2` ', text)
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
                if tag in ("replace", 'insert'):
                    delta += current_message[j1:j2]
                    
            self.previous_message = current_message
            return delta
