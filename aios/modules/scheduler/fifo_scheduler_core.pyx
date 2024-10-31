# scheduler_core.pyx
# cython: language_level=3
# distutils: language=c++

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair
from libc.stdint cimport uint32_t
from libc.time cimport time, time_t
from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref, preincrement as inc

ctypedef unsigned long size_t

cdef extern from "<algorithm>" namespace "std":
    T min[T](T a, T b) nogil

cdef struct Request:
    string request_id
    string version
    uint32_t priority
    string batch_key
    time_t timestamp
    PyObject* payload

cdef class QueueManager:
    cdef:
        unordered_map[string, unordered_map[string, vector[Request]]] _batch_queues
        size_t _max_batch_size
        size_t _min_batch_size
    
    def __cinit__(self, unsigned long max_batch_size, unsigned long min_batch_size):
        self._max_batch_size = max_batch_size
        self._min_batch_size = min_batch_size
    
    cdef vector[Request] _get_batch(self, vector[Request]* queue):
        cdef:
            vector[Request] batch
            size_t batch_size = min[size_t](queue.size(), self._max_batch_size)
            
        for i in range(batch_size):
            batch.push_back(deref(queue)[i])
        return batch
    
    cpdef bint add_request(self, str agent_name_py, str request_id_py, 
                          uint32_t priority, str batch_key_py, 
                          str version_py, object payload):
        cdef:
            string agent_name = agent_name_py.encode()
            string request_id = request_id_py.encode()
            string batch_key = batch_key_py.encode()
            string version = version_py.encode()
            Request req
        
        req.request_id = request_id
        req.version = version
        req.priority = priority
        req.batch_key = batch_key
        req.timestamp = time(NULL)
        req.payload = <PyObject*>payload
        
        self._batch_queues[agent_name][batch_key].push_back(req)
        
        return (self._batch_queues[agent_name][batch_key].size() >= 
                self._max_batch_size)
    
    cpdef list get_next_batch(self, str agent_name_py):
        cdef:
            string agent_name = agent_name_py.encode()
            vector[Request] batch
            time_t current_time = time(NULL)
            list result = []
            Request req
            unordered_map[string, vector[Request]].iterator batch_it
            vector[Request]* queue_ptr
        
        if not self._batch_queues.count(agent_name):
            return None
        
        # Iterate through batch queues using C++ iterator
        batch_it = self._batch_queues[agent_name].begin()
        while batch_it != self._batch_queues[agent_name].end():
            queue_ptr = &deref(batch_it).second
            
            if not queue_ptr.empty():
                if (queue_ptr.size() >= self._max_batch_size or
                    (queue_ptr.size() >= self._min_batch_size and
                     current_time - deref(queue_ptr)[0].timestamp >= 0.5)):
                    
                    batch = self._get_batch(queue_ptr)
                    
                    # Convert to Python list
                    for i in range(batch.size()):
                        req = batch[i]
                        result.append({
                            'request_id': req.request_id.decode(),
                            'version': req.version.decode(),
                            'priority': req.priority,
                            'batch_key': req.batch_key.decode(),
                            'payload': <object>req.payload
                        })
                    
                    # Remove processed requests
                    queue_ptr.erase(
                        queue_ptr.begin(),
                        queue_ptr.begin() + batch.size()
                    )
                    
                    return result
            
            inc(batch_it)
        
        return None
    
    cpdef void mark_batch_complete(self, str agent_name_py, list batch_ids):
        pass  # Implement if needed
    
    cpdef dict get_stats(self, str agent_name_py):
        cdef:
            string agent_name = agent_name_py.encode()
            unsigned long total_requests = 0
            unordered_map[string, vector[Request]].iterator batch_it
        
        if not self._batch_queues.count(agent_name):
            return {'total_pending': 0, 'batch_queues': 0}
        
        # Count using C++ iterator
        batch_it = self._batch_queues[agent_name].begin()
        while batch_it != self._batch_queues[agent_name].end():
            total_requests += deref(batch_it).second.size()
            inc(batch_it)
        
        return {
            'total_pending': total_requests,
            'batch_queues': self._batch_queues[agent_name].size()
        }