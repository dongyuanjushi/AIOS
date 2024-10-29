from typing import Any
from aios.modules.access.types.message import ContentType
from dataclasses import dataclass, field

@dataclass
class Content:
    type: ContentType
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "metadata": self.metadata
        }