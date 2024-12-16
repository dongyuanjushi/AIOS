from pydantic import BaseModel
from typing import Any, TypeAlias, Callable
from queue import Queue

from ..stores.queue import SignalList

StorageRequestQueue: TypeAlias = SignalList

StorageRequestQueuePopItem: TypeAlias = Callable[[], None]
StorageRequestQueueAppendItem: TypeAlias = Callable[[str], None]
StorageRequestQueueCheckEmpty: TypeAlias = Callable[[], bool]

class StorageManagerParams(BaseModel):
    root_dir: str
    use_vector_db: bool = False