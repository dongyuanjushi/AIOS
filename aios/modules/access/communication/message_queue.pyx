# message_queue.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language=c++

from cpython.ref cimport PyObject
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from libc.stdlib cimport malloc, free
import threading
import queue
import time

# Global instance storage
cdef MessageQueue _INSTANCE = None
cdef object _INSTANCE_LOCK = threading.Lock()

cdef class Message:
    cdef:
        public str msg_type
        public object data
        public double timestamp

    def __cinit__(self, str msg_type, object data):
        self.msg_type = msg_type
        self.data = data
        self.timestamp = time.time()

cdef class MessageQueue:
    cdef:
        object _subscribers
        object _queue
        bint _running
        object _processor_thread
        object _lock
        bint _initialized
    
    def __cinit__(self):
        self._subscribers = {}
        self._queue = queue.Queue()
        self._running = True
        self._initialized = False
        self._lock = threading.Lock()
        self._processor_thread = None

    def __dealloc__(self):
        self.shutdown()

    @staticmethod
    def get_instance():
        global _INSTANCE
        global _INSTANCE_LOCK
        
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = MessageQueue()
                _INSTANCE._initialized = True
                _INSTANCE._start_processor()
        return _INSTANCE

    cdef inline void _start_processor(self):
        self._processor_thread = threading.Thread(target=self._process_messages, daemon=True)
        self._processor_thread.start()

    cpdef void subscribe(self, str message_type, object callback):
        """Subscribe to a message type with a callback."""
        with self._lock:
            if message_type not in self._subscribers:
                self._subscribers[message_type] = []
            self._subscribers[message_type].append(callback)

    cpdef void unsubscribe(self, str message_type, object callback):
        """Unsubscribe from a message type."""
        with self._lock:
            if message_type in self._subscribers:
                if callback in self._subscribers[message_type]:
                    self._subscribers[message_type].remove(callback)
                if not self._subscribers[message_type]:
                    del self._subscribers[message_type]

    cpdef void emit(self, str message_type, object data=None):
        """Emit a message to subscribers."""
        cdef Message msg = Message(message_type, data)
        self._queue.put(msg)

    cdef void _process_messages(self):
        """Process messages in the queue."""
        cdef:
            Message message
            list callbacks
            object callback
        
        while self._running:
            try:
                message = self._queue.get(timeout=1.0)
                with self._lock:
                    if message.msg_type in self._subscribers:
                        # Make a copy of callbacks to avoid holding the lock
                        callbacks = self._subscribers[message.msg_type].copy()
                
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        print(f"Error in subscriber callback: {e}")
            except queue.Empty:
                continue

    cpdef void shutdown(self):
        """Shutdown the message queue."""
        if self._running:
            self._running = False
            if self._processor_thread is not None:
                self._processor_thread.join()