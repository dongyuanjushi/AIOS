# This implements a (mostly) FIFO task queue using threads and queue, in a
# similar fashion to the round robin scheduler. However, the timeout is 1 second
# instead of 0.05 seconds.

from aios.hooks.types.llm import LLMRequestQueue
from aios.hooks.types.memory import MemoryRequestQueue
from aios.hooks.types.tool import ToolRequestQueue
from aios.hooks.types.storage import StorageRequestQueue

from aios.memory.manager import MemoryManager
from aios.storage.storage import StorageManager
from aios.llm_core.adapter import LLMAdapter
from aios.tool.manager import ToolManager

from .base import Scheduler

from typing import List

from queue import Empty

import traceback
import time

class FIFOScheduler(Scheduler):
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
        super().__init__(
            llms,
            memory_manager,
            storage_manager,
            tool_manager,
            log_mode,
            # get_llm_syscall,
            # get_memory_syscall,
            # get_storage_syscall,
            # get_tool_syscall,
            llm_request_queue,
            memory_request_queue,
            storage_request_queue,
            tool_request_queue
        )
        
    def popItem(self, request_queue, index):
        while True:
            item = request_queue.pop(index)
            if item:
                return item
            time.sleep(0.1)

    def run_llm_syscall(self):
        while self.active:
            try:
                # wait at a fixed time interval, if there is nothing received in the time interval, it will raise Empty
                # llm_syscall = self.get_llm_syscall()
                llm_syscall = self.llm_request_queue.pop(0)
                
                if llm_syscall is None:
                    time.sleep(0.1)
                    continue
                
                # if llm_syscall is None:
                #     time.sleep(0.1)
                #     continue
                
                # print(f"llm_syscall is detected: {llm_syscall}")
                
                llm_syscall.set_status("executing")
                self.logger.log(
                    f"{llm_syscall.agent_name} is executing. \n", "execute"
                )
                llm_syscall.set_start_time(time.time())
                # print(self.llms)
                response = self.llms[0].address_syscall(llm_syscall)
                llm_syscall.set_response(response)

                llm_syscall.event.set()
                llm_syscall.set_status("done")
                llm_syscall.set_end_time(time.time())

            # except Empty:
            #     pass
            except Exception:
                traceback.print_exc()

    def run_memory_syscall(self):
        while self.active:
            try:
                # wait at a fixed time interval, if there is nothing received in the time interval, it will raise Empty
                memory_syscall = self.popItem(self.memory_request_queue, 0)
                
                # print(memory_syscall)
                
                memory_syscall.set_status("executing")
                self.logger.log(
                    f"{memory_syscall.agent_name} is executing. \n", "execute"
                )
                memory_syscall.set_start_time(time.time())

                response = self.memory_manager.address_request(memory_syscall)
                memory_syscall.set_response(response)

                memory_syscall.event.set()
                memory_syscall.set_status("done")
                memory_syscall.set_end_time(time.time())

            except Empty:
                pass

            except Exception:
                traceback.print_exc()

    def run_storage_syscall(self):
        while self.active:
            try:
                storage_syscall = self.popItem(self.storage_request_queue, 0)
                
                print(storage_syscall)
                
                storage_syscall.set_status("executing")
                self.logger.log(
                    f"{storage_syscall.agent_name} is executing. \n", "execute"
                )
                storage_syscall.set_start_time(time.time())

                response = self.storage_manager.address_request(storage_syscall)
                storage_syscall.set_response(response)

                storage_syscall.event.set()
                storage_syscall.set_status("done")
                storage_syscall.set_end_time(time.time())

                self.logger.log(
                    f"Current request of {storage_syscall.agent_name} is done. Thread ID is {storage_syscall.get_pid()}\n",
                    "done"
                )

            except Empty:
                pass

            except Exception:
                traceback.print_exc()

    def run_tool_syscall(self):
        while self.active:
            try:
                # tool_syscall = self.popItem(self.tool_request_queue, 0)
                tool_syscall = self.tool_request_queue.pop(0)
                
                if tool_syscall is None:
                    time.sleep(0.1)
                    continue
                # print(tool_syscall)
                
                tool_syscall.set_status("executing")

                tool_syscall.set_start_time(time.time())

                response = self.tool_manager.address_request(tool_syscall)
                tool_syscall.set_response(response)

                tool_syscall.event.set()
                tool_syscall.set_status("done")
                tool_syscall.set_end_time(time.time())

            # except Empty:
            #     pass

            except Exception:
                traceback.print_exc()
