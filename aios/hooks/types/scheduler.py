from pydantic import BaseModel
from typing import Any, TypeAlias, Callable

# from .llm import LLMRequestQueueGetMessage
# from .memory import MemoryRequestQueueGetMessage
# from .storage import StorageRequestQueueGetMessage
# from .tool import ToolRequestQueueGetMessage
from .llm import LLMRequestQueue
from .memory import MemoryRequestQueue
from .storage import StorageRequestQueue
from .tool import ToolRequestQueue

class SchedulerParams(BaseModel):
    llms: Any
    memory_manager: Any
    storage_manager: Any
    tool_manager: Any
    log_mode: str
    # get_llm_syscall: LLMRequestQueueGetMessage | None
    # get_memory_syscall: MemoryRequestQueueGetMessage | None
    # get_storage_syscall: StorageRequestQueueGetMessage | None
    # get_tool_syscall: ToolRequestQueueGetMessage | None
    llm_request_queue: LLMRequestQueue | None
    memory_request_queue: MemoryRequestQueue | None
    storage_request_queue: StorageRequestQueue | None
    tool_request_queue: ToolRequestQueue | None
    
    class Config:
        arbitrary_types_allowed = True