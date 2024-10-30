from typing import Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Agent:
    id: str  # CUID
    author: str
    name: str
    version: str
    description: str = "No Description Provided"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    system_metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs) -> None:
        """Update agent fields and automatically update the updated_at timestamp."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "author": self.author,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "system_metadata": self.system_metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        return cls(
            id=data["id"],
            author=data["author"],
            name=data["name"],
            version=data["version"],
            description=data.get("description", "No Description Provided"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            system_metadata=data.get("system_metadata", {})
        )