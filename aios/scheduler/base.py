# from aios.hooks.types.llm import LLMRequestQueueGetMessage
# from aios.hooks.types.memory import MemoryRequestQueueGetMessage
# from aios.hooks.types.tool import ToolRequestQueueGetMessage
# from aios.hooks.types.storage import StorageRequestQueueGetMessage

from aios.hooks.types.llm import LLMRequestQueue
from aios.hooks.types.memory import MemoryRequestQueue
from aios.hooks.types.storage import StorageRequestQueue
from aios.hooks.types.tool import ToolRequestQueue

from aios.utils.logger import SchedulerLogger

from abc import ABC, abstractmethod

from typing import List

from threading import Thread

from aios.memory.manager import MemoryManager
from aios.storage.storage import StorageManager
from aios.llm_core.adapter import LLMAdapter
from aios.tool.manager import ToolManager

class Scheduler:
    def __init__(
        self,
        llms: List[LLMAdapter],
        memory_manager: MemoryManager,
        storage_manager: StorageManager,
        tool_manager: ToolManager,
        log_mode,
        # get_llm_syscall: LLMRequestQueueGetMessage,
        # get_memory_syscall: MemoryRequestQueueGetMessage,
        # get_storage_syscall: StorageRequestQueueGetMessage,
        # get_tool_syscall: ToolRequestQueueGetMessage,
        llm_request_queue: LLMRequestQueue,
        memory_request_queue: MemoryRequestQueue,
        storage_request_queue: StorageRequestQueue,
        tool_request_queue: ToolRequestQueue
    ):
        # self.agent_process_queue = Queue()
        # self.get_llm_syscall = get_llm_syscall
        # self.get_memory_syscall = get_memory_syscall
        # self.get_storage_syscall = get_storage_syscall
        # self.get_tool_syscall = get_tool_syscall
        self.llm_request_queue = llm_request_queue
        self.memory_request_queue = memory_request_queue
        self.storage_request_queue = storage_request_queue
        self.tool_request_queue = tool_request_queue
        
        self.active = False  # start/stop the scheduler
        self.log_mode = log_mode
        self.logger = self.setup_logger()
        self.request_processors = {
            "llm_syscall_processor": Thread(target=self.run_llm_syscall),
            "mem_syscall_processor": Thread(target=self.run_memory_syscall),
            "sto_syscall_processor": Thread(target=self.run_storage_syscall),
            "tool_syscall_processor": Thread(target=self.run_tool_syscall),
        }
        self.llms = llms
        self.memory_manager = memory_manager
        self.storage_manager = storage_manager
        self.tool_manager = tool_manager

    def start(self):
        """start the scheduler"""
        self.active = True
        for name, thread_value in self.request_processors.items():
            thread_value.start()

    def stop(self):
        """stop the scheduler"""
        self.active = False
        for name, thread_value in self.request_processors.items():
            thread_value.join()

    def setup_logger(self):
        logger = SchedulerLogger("Scheduler", self.log_mode)
        return logger

    @abstractmethod
    def run_llm_syscall(self):
        pass
    
    @abstractmethod
    def run_memory_syscall(self):
        pass
    
    @abstractmethod
    def run_storage_syscall(self):
        pass

    @abstractmethod
    def run_tool_syscall(self):
        pass
