from functools import wraps
from flask import request, jsonify
import os

# Secret key for token validation - should be stored securely in environment variables
API_SECRET_KEY = os.getenv('API_SECRET_KEY', 'Ollama')  # Replace with secure key in production

class TokenError(Exception):
    pass

class TimeoutError(Exception):
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
