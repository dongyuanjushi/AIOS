# example.py
import asyncio
from aios.modules.scheduler.fifo import OptimizedBufferScheduler, BatchConfig, AgentRequest, Priority

async def main():
    # Configure batching
    batch_config = BatchConfig(
        max_batch_size=3,     # Small for demonstration
        wait_time_seconds=1.0, # Short wait time for demonstration
        min_batch_size=2      # Process when at least 2 requests available
    )
    
    # Create scheduler
    scheduler = OptimizedBufferScheduler(batch_config)
    
    # Add batch execution handler
    async def handle_batch(data):
        agent_name = data['agent_name']
        batch = data['batch']
        
        print(f"\nProcessing batch for {agent_name}:")
        for request in batch:
            print(f"- Request {request['request_id']}: {request['payload']}")
        
        # Mark batch complete
        batch_ids = [req['request_id'] for req in batch]
        await scheduler.mark_batch_complete(agent_name, batch_ids)
        
        # Show queue stats after processing
        stats = scheduler.get_queue_stats(agent_name)
        print(f"Remaining requests: {stats['total_pending']}")
    
    scheduler.emitter.on('execute_batch', handle_batch)
    
    # Create and schedule some requests
    print("Scheduling requests...")
    requests = [
        AgentRequest(
            agent_name="agent1",
            version="1.0",
            batch_key="batch1",
            priority=Priority.MEDIUM,
            payload={"data": f"request_{i}"}
        )
        for i in range(5)  # Create 5 requests
    ]
    
    for request in requests:
        await scheduler.schedule(request)
        print(f"Scheduled request {request.request_id}")
    
    # Keep running to see batches process
    print("\nWaiting for batches to process...")
    await asyncio.sleep(5)  # Give time for batches to process
    
    # Shutdown gracefully
    await scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())