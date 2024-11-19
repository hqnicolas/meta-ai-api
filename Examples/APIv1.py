from flask import Flask, request, Response, jsonify
from meta_ai_api import MetaAI
from difflib import SequenceMatcher
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any

app = Flask(__name__)

class ChatRequest:
    def __init__(self, 
                 messages: List[Dict[str, str]], 
                 model: str = 'llama3.2:70b-text-fp16',
                 stream: bool = False):
        self.messages = messages
        self.model = model
        self.stream = stream

class ChatResponse:
    def __init__(self, 
                 id: str,
                 model: str,
                 choices: List[Dict[str, Any]],
                 created: int,
                 usage: Optional[Dict[str, int]] = None):
        self.id = id
        self.model = model
        self.choices = choices
        self.created = created
        self.usage = usage

    def to_dict(self) -> Dict[str, Any]:
        response = {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": self.choices
        }
        if self.usage:
            response["usage"] = self.usage
        return response

class ChatStreamChunk:
    def __init__(self,
                 id: str,
                 model: str,
                 delta_content: Optional[str] = None,
                 finish_reason: Optional[str] = None):
        self.id = id
        self.model = model
        self.delta_content = delta_content
        self.finish_reason = finish_reason
        self.created = int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        choices = [{
            "index": 0,
            "finish_reason": self.finish_reason
        }]

        if self.delta_content is not None:
            choices[0]["delta"] = {"content": self.delta_content}
        else:
            choices[0]["delta"] = {}

        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices
        }

class ResponseWriter:
    def __init__(self):
        self.previous_message = ""

    def normalize_text(self, text: str) -> str:
        """Normalize text while preserving spacing"""
        text = text.replace('\\"', '\\\" ')
        text = text.replace(':\"', ':\" ')
        text = text.replace('""""', '###')
        text = text.replace('"""', '```')
        text = text.replace('":', '" :')
        return text

    def get_delta_content(self, current_message: str, threshold: float = 0.6) -> str:
        """Calculate delta between previous and current message"""
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

def generate_chat_id() -> str:
    """Generate a unique chat completion ID"""
    return f"chatcmpl-{str(uuid.uuid4())}"

def authenticate_request(request) -> bool:
    """Authenticate incoming requests"""
    authenticated_routes = ['/chat/completions', '/api/chat']
    
    if request.path in authenticated_routes:
        return True

    auth_header = request.headers.get('Authorization')
    return bool(auth_header and auth_header.startswith('Bearer Ollama'))

@app.route('/models', methods=['POST', 'GET'])
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

@app.route('/chat/completions', methods=['POST', 'GET'])
@app.route('/api/chat', methods=['POST', 'GET'])
def handle_chat_completion():
    """Handle chat completion requests"""
    if not authenticate_request(request):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        chat_request = ChatRequest(
            messages=data.get('messages', []),
            model=data.get('model', 'llama3.2:70b-text-fp16'),
            stream=data.get('stream', False)
        )

        ai = MetaAI()
        response = ai.prompt(message="{0}".format(chat_request.messages), stream=chat_request.stream)

        if chat_request.stream:
            return Response(
                stream_response(response, chat_request.model),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
        else:
            return handle_sync_response(response, chat_request.model)

    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "param": None,
                "code": "internal_error"
            }
        }), 500

def stream_response(response: Generator, model: str) -> Generator:
    """Handle streaming response generation"""
    writer = ResponseWriter()
    
    try:
        for chunk in response:
            if isinstance(chunk, dict) and 'message' in chunk:
                current_message = chunk['message'].strip()
                delta_content = writer.get_delta_content(current_message)
                
                if delta_content:
                    delta_content = writer.normalize_text(delta_content)
                    stream_chunk = ChatStreamChunk(
                        id=generate_chat_id(),
                        model=model,
                        delta_content=delta_content
                    )
                    yield f"data: {json.dumps(stream_chunk.to_dict())}\n\n"
                    
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
    finally:
        final_chunk = ChatStreamChunk(
            id=generate_chat_id(),
            model=model,
            finish_reason="stop"
        )
        yield f"data: {json.dumps(final_chunk.to_dict())}\n\n"
        yield "data: [DONE]\n\n"

def handle_sync_response(response: Dict[str, Any], model: str) -> Response:
    """Handle synchronous response generation"""
    if isinstance(response, dict) and 'message' in response:
        response_content = response['message']
    else:
        response_content = str(response)

    chat_response = ChatResponse(
        id=generate_chat_id(),
        model=model,
        created=int(time.time()),
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(str(response)),
            "completion_tokens": len(response_content),
            "total_tokens": len(str(response)) + len(response_content)
        }
    )

    return jsonify(chat_response.to_dict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)
