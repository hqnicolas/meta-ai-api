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
import hmac
import hashlib
import os
from base64 import b64encode

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)
request_timeout = 30

# Secret key for token validation - should be stored securely in environment variables
API_SECRET_KEY = os.getenv('API_SECRET_KEY', 'Ollama')  # Replace with secure key in production

class TokenError(Exception):
    pass

def generate_token(key: str = API_SECRET_KEY) -> str:
    """Generate a secure API token"""
    timestamp = str(int(time.time()))
    random_bytes = os.urandom(16)
    message = timestamp.encode() + random_bytes
    signature = hmac.new(key.encode(), message, hashlib.sha256).digest()
    token = b64encode(message + signature).decode()
    return token

def verify_token(token: str, key: str = API_SECRET_KEY) -> bool:
    """Verify the API token"""
    try:
        # Decode the token
        decoded = b64encode(token.encode()).decode()
        message = decoded[:-32]  # Extract message (timestamp + random bytes)
        received_signature = decoded[-32:]  # Extract signature
        
        # Verify signature
        expected_signature = hmac.new(key.encode(), message.encode(), hashlib.sha256).digest()
        return hmac.compare_digest(received_signature, expected_signature)
    except Exception:
        return False

def require_bearer_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                "error": {
                    "message": "No authorization header",
                    "type": "authentication_error",
                    "code": "no_auth_header"
                }
            }), 401
            
        try:
            # Extract token from "Bearer <token>"
            if not auth_header.startswith('Bearer '):
                raise TokenError("Invalid authorization header format")
                
            token = auth_header.split(' ')[1]
            
            if not token:
                raise TokenError("Empty token")
                
            if not verify_token(token):
                raise TokenError("Invalid token")
                
            return f(*args, **kwargs)
            
        except TokenError as e:
            return jsonify({
                "error": {
                    "message": str(e),
                    "type": "authentication_error",
                    "code": "invalid_token"
                }
            }), 401
        except Exception as e:
            return jsonify({
                "error": {
                    "message": "Authentication error",
                    "type": "internal_error",
                    "code": "auth_error"
                }
            }), 500
            
    return decorated

class TimeoutError(Exception):
    pass

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

class ChatHandler:
    def __init__(self, model: str, messages: List[Dict[str, str]], stream: bool = False):
        self.model = model
        self.messages = self._format_messages(messages)
        self.stream = stream
        self.ai = MetaAI()
        self.writer = ResponseWriter()

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single string for the AI model"""
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if isinstance(content, list):
                # Handle continue.dev's array format
                content = ' '.join(item.get('text', '') for item in content if item.get('type') == 'text')
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _create_stream_chunk(self, delta_content: Optional[str] = None, finish_reason: Optional[str] = None) -> Dict:
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

def require_bearer_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                "error": {
                    "message": "No authorization header",
                    "type": "authentication_error",
                    "code": "no_auth_header"
                }
            }), 401
            
        try:
            # Support both Bearer token and direct API key
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                if not verify_token(token):
                    # If token verification fails, check if it matches the API key
                    if token != os.getenv('API_KEY', 'Ollama'):
                        raise TokenError("Invalid token")
            else:
                # Direct API key usage
                if auth_header != os.getenv('API_KEY', 'Ollama'):
                    raise TokenError("Invalid API key")
                
            return f(*args, **kwargs)
            
        except TokenError as e:
            return jsonify({
                "error": {
                    "message": str(e),
                    "type": "authentication_error",
                    "code": "invalid_token"
                }
            }), 401
        except Exception as e:
            return jsonify({
                "error": {
                    "message": "Authentication error",
                    "type": "internal_error",
                    "code": "auth_error"
                }
            }), 500
            
    return decorated

@app.route('/', methods=['GET', 'HEAD'])
def get_status():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/auth/token', methods=['POST'])
def get_token():
    """Endpoint to get an API token"""
    api_key = request.headers.get('X-API-Key')
    
    if not api_key or api_key != os.getenv('API_KEY', 'Ollama'):  # Replace with secure API key
        return jsonify({
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error",
                "code": "invalid_api_key"
            }
        }), 401
        
    token = generate_token()
    return jsonify({
        "token": token,
        "expires_at": int(time.time()) + 3600  # Token expires in 1 hour
    })

@app.route('/api/generate/<int:url_idx>', methods=['POST'])
@app.route('/api/generate', methods=['POST'])
@require_bearer_token
def generate_completion(url_idx=None):
    """Handle text completion requests"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    try:
        data = request.get_json()
        handler = ChatHandler(
            model=data.get('model', 'llama3.2:70b-text-fp16'),
            messages=[{"role": "user", "content": data.get('prompt', '')}],
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

    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        }), 500

@app.route('/api/chat/<int:url_idx>', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
@require_bearer_token
def generate_chat_completion(url_idx=None):
    """Handle chat completion requests"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    bypass_filter = request.args.get('bypass_filter', 'false').lower() == 'true'
    
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

    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        }), 500

@app.route('/v1/chat/completions', methods=['POST'])
def generate_openai_chat_completion():
    """Handle OpenAI-compatible chat completion requests"""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    # Extract API key from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        api_key = request.headers.get('api-key')  # Try alternative header
        if not api_key:
            return jsonify({"error": "Missing authentication"}), 401
    else:
        # Handle "Bearer <token>" format
        api_key = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else auth_header

    # Verify API key
    if api_key != os.getenv('API_KEY', 'Ollama'):
        return jsonify({"error": "Invalid authentication"}), 403

    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        # Convert messages to the format expected by MetaAI
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                # Handle continue.dev's array format
                content = ' '.join(item.get('text', '') for item in msg['content'] if item.get('type') == 'text')
            else:
                content = msg.get('content', '')
            
            formatted_messages.append({
                'role': msg.get('role', 'user'),
                'content': content
            })

        handler = ChatHandler(
            model=data.get('model', 'llama3.2:70b-text-fp16'),
            messages=formatted_messages,
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

    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        }), 500

# Add CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,api-key')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Handle OPTIONS requests
@app.route('/v1/chat/completions', methods=['OPTIONS'])
def options():
    return '', 200

@app.route('/v1/models/<int:url_idx>', methods=['GET'])
@app.route('/v1/models', methods=['GET'])
@require_bearer_token
def get_openai_models(url_idx=None):
    """Handle model listing endpoint with OpenAI compatibility"""
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