from flask import Flask, request, Response, jsonify
from meta_ai_api import MetaAI
from difflib import SequenceMatcher
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import asyncio

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)
request_timeout = 30  # 30 seconds timeout

class TimeoutError(Exception):
    pass

class ResponseWriter:
    def __init__(self):
        self.previous_message = ""
        self._lock = threading.Lock()

    def normalize_text(self, text: str) -> str:
        """Normalize text while preserving spacing"""
        text = text.replace('\\"', '\\\" ')
        text = text.replace(':\"', ':\" ')
        text = text.replace('""""', '###')
        text = text.replace('"""', '```')
        text = text.replace('":', '" :')
        return text

    def get_delta_content(self, current_message: str, threshold: float = 0.6) -> str:
        """Calculate delta between previous and current message with thread safety"""
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

class ChatHandler:
    def __init__(self, model: str, messages: List[Dict[str, str]], stream: bool = False):
        self.model = model
        self.messages = messages
        self.stream = stream
        self.ai = MetaAI()
        self.writer = ResponseWriter()

    def _create_stream_chunk(self, delta_content: Optional[str] = None, finish_reason: Optional[str] = None) -> Dict:
        """Create a stream chunk response"""
        choices = [{
            "index": 0,
            "finish_reason": finish_reason
        }]

        if delta_content is not None:
            choices[0]["delta"] = {"content": delta_content}
        else:
            choices[0]["delta"] = {}

        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "choices": choices
        }

    def process_stream(self) -> Generator[str, None, None]:
        try:
            response = self.ai.prompt(message="{0}".format(self.messages), stream=True)
            
            for chunk in response:
                if isinstance(chunk, dict) and 'message' in chunk:
                    current_message = chunk['message'].strip()
                    delta_content = self.writer.get_delta_content(current_message)
                    
                    if delta_content:
                        delta_content = self.writer.normalize_text(delta_content)
                        stream_chunk = self._create_stream_chunk(delta_content=delta_content)
                        yield f"data: {json.dumps(stream_chunk)}\n\n"

            # Send final chunk
            final_chunk = self._create_stream_chunk(finish_reason="stop")
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    def process_sync(self) -> Dict[str, Any]:
        """Process synchronous response"""
        try:
            response = self.ai.prompt(
                message="{0}".format(self.messages),
                stream=False
            )
            
            if isinstance(response, dict) and 'message' in response:
                response_content = response['message']
            else:
                response_content = str(response)

            return {
                "id": f"chatcmpl-{str(uuid.uuid4())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(str(response)),
                    "completion_tokens": len(response_content),
                    "total_tokens": len(str(response)) + len(response_content)
                }
            }

        except Exception as e:
            return {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            future = executor.submit(f, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError:
                raise TimeoutError("Request timed out")
        return wrapped
    return decorator

@app.route('/chat/completions', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
def handle_chat_completion():
    """Handle chat completion requests"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    try:
        data = request.get_json()
        handler = ChatHandler(
            model=data.get('model', 'llama3.2:70b-text-fp16'),
            messages=data.get('messages', []),
            stream=data.get('stream', False)
        )

        if data.get('stream', False):
            return Response(
                handler.process_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
        else:
            response = handler.process_sync()
            if 'error' in response:
                return jsonify(response), 500
            return jsonify(response)

    except TimeoutError as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "timeout_error",
                "code": "timeout_error"
            }
        }), 504
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Handle model listing endpoint"""
    models = {
        "data": [{
            "id": "llama3.2:70b-text-fp16",
            "object": "model",
            "created": 23102024,
            "owned_by": "meta.ai",
            "permission": [{
                "id": "llama3.2-70b",
                "object": "model_permission",
                "created": 23102024,
                "allow_create_engine": True
            }]
        }],
        "object": "list"
    }
    return jsonify(models)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True, threaded=True)
