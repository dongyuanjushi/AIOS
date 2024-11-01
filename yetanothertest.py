from aios.modules.access.communication.message_queue import MessageQueue
import time  # Add this import

def handle_message(message):
    print(f"Received: {message.data}")

# Get singleton instance
mq = MessageQueue.get_instance()

# Subscribe to messages
mq.subscribe("example_type", handle_message)

# Emit messages
mq.emit("example_type", {"data": "test"})

# Add a small delay to allow message processing
time.sleep(0.1)  # Wait for 100ms