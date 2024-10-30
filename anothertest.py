from openai import OpenAI
from aios.modules.access.conversation import Conversation

from dotenv import load_dotenv
load_dotenv()

conversation = Conversation()
conversation.add_user_message("Hello!")

# Synchronous call
client = OpenAI()

response = conversation.send_to_openai_sync(
    client,
    model="gpt-4",
    temperature=0.7
)

print(response)

print(conversation)