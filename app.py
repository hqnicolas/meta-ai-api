from flask import Flask, request, Response, jsonify
from meta_ai_api import MetaAI
import time
from typing import Dict, List, Optional, Generator, Any
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from auth.decorators import require_bearer_token, generate_token
from handlers.chat import ChatHandler

import os
from base64 import b64encode

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)
request_timeout = 30

# Secret key for token validation - should be stored securely in environment variables
API_SECRET_KEY = os.getenv('API_SECRET_KEY', 'Ollama')  # Replace with secure key in production

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