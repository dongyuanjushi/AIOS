# scheduler.py
from dataclasses import dataclass
import asyncio
from typing import Optional, Any, Dict
from enum import IntEnum
from pyee import AsyncIOEventEmitter
import uuid
from fifo_scheduler_core import QueueManager  # Cython-optimized part

class Priority(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class BatchConfig:
    max_batch_size: int = 10
    wait_time_seconds: float = 0.5
    min_batch_size: int = 1

@dataclass
class AgentRequest:
    agent_name: str
    version: str
    priority: Priority = Priority.MEDIUM
    batch_key: Optional[str] = None
    request_id: str = None
    payload: Any = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

class OptimizedBufferScheduler:
    """
    Pure scheduler that manages queuing and emitting of agent requests with batching support.
    Uses Cython-optimized queue management for performance.
    """
    def __init__(self, batch_config: Optional[BatchConfig] = None):
        self._event_loop = asyncio.get_event_loop()
        self.emitter = AsyncIOEventEmitter()
        self.batch_config = batch_config or BatchConfig()
        
        # Initialize Cython-optimized queue manager with batch settings
        self._queue_manager = QueueManager(
            self.batch_config.max_batch_size,
            self.batch_config.min_batch_size
        )
        
        # Track batch processors
        self._batch_tasks: Dict[str, asyncio.Task] = {}

    async def schedule(self, request: AgentRequest) -> None:
        """Schedule a request with optional batching."""
        # Add to Cython-managed queues
        batch_ready = self._queue_manager.add_request(
            request.agent_name,
            request.request_id,
            request.priority,
            request.batch_key or "_default",
            request.version,
            request.payload
        )

        # Ensure batch processor is running
        await self._ensure_batch_processor(request.agent_name)

    async def _ensure_batch_processor(self, agent_name: str) -> None:
        """Ensure a batch processor exists for this agent."""
        if (agent_name not in self._batch_tasks or 
            self._batch_tasks[agent_name].done()):
            self._batch_tasks[agent_name] = asyncio.create_task(
                self._process_agent_batches(agent_name)
            )

    async def _process_agent_batches(self, agent_name: str) -> None:
        """Continuously process batches for an agent."""
        while True:
            # Get batch from Cython queue manager
            batch = self._queue_manager.get_next_batch(agent_name)
            
            if batch:
                # Emit batch for processing
                await self.emitter.emit('execute_batch', {
                    'agent_name': agent_name,
                    'batch': batch
                })
            
            # Wait before next check
            await asyncio.sleep(self.batch_config.wait_time_seconds)

    async def mark_batch_complete(self, agent_name: str, batch_ids: list) -> None:
        """Mark a batch as complete."""
        self._queue_manager.mark_batch_complete(agent_name, batch_ids)

    def get_queue_stats(self, agent_name: str) -> dict:
        """Get queue statistics."""
        return self._queue_manager.get_stats(agent_name)