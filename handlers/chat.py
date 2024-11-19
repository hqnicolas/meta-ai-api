import json
import time
import uuid
from typing import Dict, List, Optional, Generator, Any
from functools import wraps
from base64 import b64encode
from meta_ai_api import MetaAI
from utils.response_writer import ResponseWriter

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