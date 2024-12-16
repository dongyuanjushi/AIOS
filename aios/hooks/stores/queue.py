# from queue import Queue

from typing import List, Callable, Optional

import threading

class SignalList:
    def __init__(self):
        self._list: List = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
    def append(self, item):
        with self._lock:
            self._list.append(item)
            self._condition.notify()
    
    def insert(self, index: int, item):
        with self._lock:
            self._list.insert(index, item)
            self._condition.notify()
    
    def remove(self, item) -> bool:
        with self._lock:
            try:
                self._list.remove(item)
                return True
            except ValueError:
                return False
    
    def pop(self, index: int = -1):
        with self._lock:
            while not self._list:
                self._condition.wait()
            return self._list.pop(index)
    
    def sort(self, key: Callable = None, reverse: bool = False):
        with self._lock:
            self._list.sort(key=key, reverse=reverse)
    
    # def get_next(self) -> Optional[Request]:
    #     with self._condition:
    #         while not self._list:
    #             self._condition.wait()
    #         return self._list.pop(0) if self._list else None
    
    def peek(self):
        with self._lock:
            return self._list[0] if self._list else None
    
    # def get_by_predicate(self, predicate: Callable[[Request], bool]) -> Optional[Request]:
    #     with self._lock:
    #         for i, item in enumerate(self._list):
    #             if predicate(item):
    #                 return self._list.pop(i)
    #         return None
    
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._list) == 0
    
    def clear(self):
        with self._lock:
            self._list.clear()
    
    def __len__(self):
        with self._lock:
            return len(self._list)
        
REQUEST_QUEUE: dict[str, SignalList] = {}

def getItem(q: SignalList):
    # return q.get(block=True, timeout=0.1)
    return q.pop(0)

def addItem(q: SignalList, item):
    # q.put(message)
    q.append(item)

    return None

def isEmpty(q: SignalList):
    # return q.empty()
    return len(q) == 0
