from typing import Tuple

from aios.hooks.types.storage import (
    StorageRequestQueue,
    StorageRequestQueuePopItem,
    StorageRequestQueueAppendItem,
    StorageRequestQueueCheckEmpty
)
from aios.hooks.utils.validate import validate
from aios.hooks.stores import queue as QueueStore
from aios.storage.storage import StorageManager

from aios.hooks.types.storage import StorageManagerParams

def useStorageRequestQueue() -> (
    Tuple[StorageRequestQueue, StorageRequestQueuePopItem, StorageRequestQueueAppendItem, StorageRequestQueueCheckEmpty]
):
    """
    Creates and returns a queue for Storage-related requests along with helper methods to manage the queue.

    Returns:
        Tuple: A tuple containing the Memory request queue, get message function, add message function, and check queue empty function.
    """
    # r_str = (
    #     generate_random_string()
    # )  # Generate a random string for queue identification
    r_str = "storage"
    _ = StorageRequestQueue()

    # Store the LLM request queue in QueueStore
    QueueStore.REQUEST_QUEUE[r_str] = _

    # Function to get messages from the queue
    def popItem():
        return QueueStore.popItem(_)

    # Function to add messages to the queue
    def appendItem(item):
        return QueueStore.appendItem(_, item)

    # Function to check if the queue is empty
    def isEmpty():
        return QueueStore.isEmpty(_)

    return _, popItem, appendItem, isEmpty

@validate(StorageManagerParams)
def useStorageManager(params: StorageManagerParams) -> StorageManager:
    return StorageManager(**params.model_dump())