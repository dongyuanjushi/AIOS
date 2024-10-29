from dataclasses import dataclass, field

from typing import Optional, Any
from datetime import UTC, datetime
import json
from uuid import uuid4

from aios.modules.access.content import Content
from aios.modules.access.tool_call import ToolCall
from aios.modules.access.types.message import ContentType, MessageRole




@dataclass
class MultimodalMessage:
    role: MessageRole
    contents: list[Content]
    message_id: str = field(default_factory=lambda: str(uuid4()))
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_text(self, text: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Add text content to the message."""
        self.contents.append(Content(
            type=ContentType.TEXT,
            data=text,
            metadata=metadata or {}
        ))

    def add_image(self, image_data: bytes, mime_type: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Add image content to the message."""
        self.contents.append(Content(
            type=ContentType.IMAGE,
            data=image_data,
            metadata={"mime_type": mime_type, **(metadata or {})}
        ))

    def add_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Add a tool call to the message."""
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments)
        self.contents.append(Content(
            type=ContentType.TOOL_CALL,
            data=tool_call.to_dict()
        ))

    def add_tool_response(self, tool_name: str, response: Any, metadata: Optional[dict[str, Any]] = None) -> None:
        """Add a tool response to the message."""
        self.contents.append(Content(
            type=ContentType.TOOL_RESPONSE,
            data={"tool_name": tool_name, "response": response},
            metadata=metadata or {}
        ))

    def get_text_contents(self) -> list[str]:
        """Get all text contents from the message."""
        return [
            content.data 
            for content in self.contents 
            if content.type == ContentType.TEXT
        ]

    def get_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls from the message."""
        return [
            ToolCall(**content.data)
            for content in self.contents
            if content.type == ContentType.TOOL_CALL
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a dictionary format."""
        return {
            "message_id": self.message_id,
            "parent_id": self.parent_id,
            "role": self.role.value,
            "contents": [content.to_dict() for content in self.contents],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert the message to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultimodalMessage":
        """Create a MultimodalMessage instance from a dictionary."""
        contents = [
            Content(
                type=ContentType(content["type"]),
                data=content["data"],
                metadata=content.get("metadata", {})
            )
            for content in data["contents"]
        ]
        
        return cls(
            role=MessageRole(data["role"]),
            contents=contents,
            message_id=data.get("message_id", str(uuid4())),
            parent_id=data.get("parent_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MultimodalMessage":
        """Create a MultimodalMessage instance from a JSON string."""
        return cls.from_dict(json.loads(json_str))