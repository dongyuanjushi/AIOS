from pydantic import BaseModel
from typing import Any, TypeAlias, Callable, List
from queue import Queue

from ..stores.queue import SignalList

MemoryRequestQueue: TypeAlias = SignalList

MemoryRequestQueuePopItem: TypeAlias = Callable[[], None]
MemoryRequestQueueAppendItem: TypeAlias = Callable[[str], None]
MemoryRequestQueueCheckEmpty: TypeAlias = Callable[[], bool]

class MemoryManagerParams(BaseModel):
    memory_limit: int
    eviction_k: int
    storage_manager: Any