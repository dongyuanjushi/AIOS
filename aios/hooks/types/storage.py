from pydantic import BaseModel
from typing import Any, TypeAlias, Callable
from queue import Queue

StorageRequestQueue: TypeAlias = Queue

StorageRequestQueueGetMessage: TypeAlias = Callable[[], None]
StorageRequestQueueAddMessage: TypeAlias = Callable[[str], None]
StorageRequestQueueCheckEmpty: TypeAlias = Callable[[], bool]