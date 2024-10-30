from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "created_at": self.created_at.isoformat()
        }
