from pydantic import BaseModel
from typing import Any, TypeAlias, Callable
from queue import Queue

from ..stores.queue import SignalList

LLMRequestQueue: TypeAlias = SignalList

LLMRequestQueuePopItem: TypeAlias = Callable[[], None]
LLMRequestQueueAppendItem: TypeAlias = Callable[[str], None]
LLMRequestQueueCheckEmpty: TypeAlias = Callable[[], bool]


class LLMParams(BaseModel):
    llm_name: str
    max_gpu_memory: dict | None = (None,)
    eval_device: str | None = (None,)
    max_new_tokens: int = (256,)
    log_mode: str = ("console",)
    llm_backend: str | None = None
