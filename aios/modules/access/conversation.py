from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json
from datetime import datetime
from base64 import b64encode

from aios.modules.access.content import Content
from aios.modules.access.message import MultimodalMessage
from aios.modules.access.types.message import ContentType, MessageRole

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

@dataclass
class Conversation:
    messages: List[MultimodalMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, message: MultimodalMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def add_system_message(self, text: str) -> None:
        """Add a system message to the conversation."""
        message = MultimodalMessage(role=MessageRole.SYSTEM, contents=[])
        message.add_text(text)
        self.add_message(message)

    def add_user_message(self, text: Optional[str] = None, image_data: Optional[bytes] = None, 
                        image_mime_type: Optional[str] = None) -> None:
        """Add a user message with optional text and image."""
        message = MultimodalMessage(role=MessageRole.USER, contents=[])
        if text:
            message.add_text(text)
        if image_data and image_mime_type:
            message.add_image(image_data, image_mime_type)
        self.add_message(message)

    def add_assistant_message(self, text: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add an assistant message with optional tool calls."""
        message = MultimodalMessage(role=MessageRole.ASSISTANT, contents=[])
        message.add_text(text)
        if tool_calls:
            for tool_call in tool_calls:
                message.add_tool_call(tool_call["name"], tool_call["arguments"])
        self.add_message(message)

    def add_tool_message(self, tool_name: str, response: Any) -> None:
        """Add a tool response message."""
        message = MultimodalMessage(role=MessageRole.TOOL, contents=[])
        message.add_tool_response(tool_name, response)
        self.add_message(message)

    def _convert_content_to_openai_format(self, content: Content) -> Union[str, Dict[str, Any]]:
        """Convert a Content object to OpenAI's format."""
        if content.type == ContentType.TEXT:
            return {"type": "text", "text": content.data}
        
        elif content.type == ContentType.IMAGE:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content.metadata.get('mime_type', 'image/jpeg')};base64,{b64encode(content.data).decode('utf-8')}"
                }
            }
        
        elif content.type == ContentType.TOOL_CALL:
            tool_call = content.data
            return {
                "type": "function",
                "function": {
                    "name": tool_call["tool_name"],
                    "arguments": json.dumps(tool_call["arguments"])
                }
            }
        
        elif content.type == ContentType.TOOL_RESPONSE:
            return {
                "type": "text",
                "text": json.dumps(content.data["response"])
            }
        
        else:
            raise ValueError(f"Unsupported content type: {content.type}")

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """Convert the conversation to OpenAI's message format."""
        openai_messages = []
        
        for message in self.messages:
            if not message.contents:
                continue

            openai_message = {"role": message.role.value}
            
            # Handle content based on OpenAI's format
            if message.role == MessageRole.TOOL:
                # Tool messages in OpenAI format
                tool_content = message.contents[0]
                if tool_content.type == ContentType.TOOL_RESPONSE:
                    openai_message["tool_call_id"] = tool_content.metadata.get("tool_call_id", "unknown")
                    openai_message["content"] = json.dumps(tool_content.data["response"])
                    openai_message["name"] = tool_content.data["tool_name"]
            else:
                # Handle regular messages with potential multiple contents
                contents = [self._convert_content_to_openai_format(content) 
                          for content in message.contents]
                
                # If there's only one text content, use simple format
                if len(contents) == 1 and isinstance(contents[0], dict) and contents[0]["type"] == "text":
                    openai_message["content"] = contents[0]["text"]
                else:
                    openai_message["content"] = contents

            openai_messages.append(openai_message)

        return openai_messages

    def create_openai_request(self, model: str = "gpt-4-vision-preview", 
                            max_tokens: int = 1000,
                            temperature: float = 1.0,
                            **kwargs) -> Dict[str, Any]:
        """Create a complete request body for OpenAI's API."""
        return {
            "model": model,
            "messages": self.to_openai_messages(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

    def send_to_openai_sync(self, client: OpenAI, **kwargs) -> Dict[str, Any]:
        """
        Send the conversation to OpenAI's API using a synchronous client.
        
        Args:
            client: An initialized OpenAI client (synchronous)
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The OpenAI API response
            
        Example:
            from openai import OpenAI
            client = OpenAI(api_key="your-api-key")
            response = conversation.send_to_openai_sync(client)
        """
        request_body = self.create_openai_request(**kwargs)
        response: ChatCompletion  = client.chat.completions.create(**request_body)
        
        # Add the assistant's response to the conversation
        assistant_message = response.choices[0].message

        self.add_assistant_message(
            text=assistant_message.content,
            tool_calls=[{
                "name": tool.function.name,
                "arguments": json.loads(tool.function.arguments)
            } for tool in (assistant_message.tool_calls or [])]
        )
        
        return response

    async def send_to_openai_async(self, client: AsyncOpenAI, **kwargs) -> Dict[str, Any]:
        """
        Send the conversation to OpenAI's API using an async client.
        
        Args:
            client: An initialized AsyncOpenAI client
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The OpenAI API response
            
        Example:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key="your-api-key")
            response = await conversation.send_to_openai_async(client)
        """
        request_body = self.create_openai_request(**kwargs)
        response: ChatCompletion = await client.chat.completions.create(**request_body)
        
        # Add the assistant's response to the conversation
        assistant_message = response.choices[0].message

        self.add_assistant_message(
            text=assistant_message.content,
            tool_calls=[{
                "name": tool.function.name,
                "arguments": json.loads(tool.function.arguments)
            } for tool in (assistant_message.tool_calls or [])]
        )
        
        return response

    # Alias for backward compatibility
    send_to_openai = send_to_openai_async

    def save_to_json(self, filepath: str) -> None:
        """Save the conversation to a JSON file."""
        data = {
            "messages": [message.to_dict() for message in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "Conversation":
        """Load a conversation from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        conversation = cls(
            messages=[MultimodalMessage.from_dict(msg) for msg in data["messages"]],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
        return conversation
    
