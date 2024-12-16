from pydantic import BaseModel
from typing import Any, TypeAlias, Callable
from queue import Queue

from ..stores.queue import SignalList

ToolRequestQueue: TypeAlias = SignalList

ToolRequestQueuePopItem: TypeAlias = Callable[[], None]
ToolRequestQueueAppendItem: TypeAlias = Callable[[str], None]
ToolRequestQueueCheckEmpty: TypeAlias = Callable[[], bool]

class ToolManagerParams(BaseModel):
    name: str
    params: dict | None = (None,)