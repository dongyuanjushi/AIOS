from typing import  Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from copy import deepcopy

from aios.modules.access.agent import Agent
from aios.modules.access.conversation import Conversation
from aios.modules.access.message import MultimodalMessage
from aios.modules.access.types.message import ContentType, MessageRole

@dataclass
class MultiAgentConversationManager:
    agents: Dict[str, Agent] = field(default_factory=dict)
    conversations: Dict[str, Conversation] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_agent(self, 
                 id: str,
                 author: str,
                 name: str,
                 version: str,
                 description: Optional[str] = None,
                 license: Optional[str] = None,
                 entry: Optional[str] = None,
                 module: Optional[str] = None,
                 **additional_metadata) -> Agent:
        """
        Add a new agent to the system.
        
        Args:
            id: CUID for the agent
            author: Author of the agent
            name: Name of the agent
            version: Version string
            description: Optional description
            license: Optional license information
            entry: Optional entry point
            module: Optional module information
            **additional_metadata: Any additional system metadata
        """
        # Combine all system metadata
        system_metadata = {
            **({"license": license} if license else {}),
            **({"entry": entry} if entry else {}),
            **({"module": module} if module else {}),
            **additional_metadata
        }

        agent = Agent(
            id=id,
            author=author,
            name=name,
            version=version,
            description=description or "No Description Provided",
            system_metadata=system_metadata
        )
        
        self.agents[id] = agent
        self.conversations[id] = Conversation()
        return agent

    def get_agent(self, agent_id: str) -> Agent:
        """Get an agent by ID."""
        if agent_id not in self.agents:
            raise KeyError(f"Agent with ID '{agent_id}' not found")
        return self.agents[agent_id]

    def get_conversation(self, agent_id: str) -> Conversation:
        """Get the conversation for a specific agent by ID."""
        if agent_id not in self.conversations:
            raise KeyError(f"No conversation found for agent ID '{agent_id}'")
        return self.conversations[agent_id]

    def add_message_to_agent(self, agent_id: str, message: MultimodalMessage) -> None:
        """Add a message to a specific agent's conversation using agent ID."""
        if agent_id not in self.conversations:
            raise KeyError(f"Agent with ID '{agent_id}' not found")
        self.conversations[agent_id].add_message(message)

    def create_unified_conversation(self, 
                                 agent_ids: Optional[Set[str]] = None,
                                 include_agent_descriptions: bool = True,
                                 include_agent_names: bool = True,
                                 agent_name_format: str = "[{agent_name}]: {content}",
                                 system_message_override: Optional[str] = None) -> Conversation:
        """
        Create a unified conversation combining messages from multiple agents.
        
        Args:
            agent_ids: Set of agent IDs to include. If None, includes all agents.
            include_agent_descriptions: Whether to include agent descriptions in the system message.
            include_agent_names: Whether to prefix messages with agent names.
            agent_name_format: Format string for adding agent names to messages.
            system_message_override: Optional system message to use instead of the default.
        """
        unified_conv = Conversation()
        
        # Determine which agents to include
        agents_to_include = set(agent_ids or self.agents.keys())
        
        # Create system message
        if system_message_override:
            system_text = system_message_override
        elif include_agent_descriptions:
            agents_desc = "\n".join([
                f"- {agent.name} (by {agent.author}, v{agent.version}): {agent.description}"
                for id, agent in self.agents.items()
                if id in agents_to_include
            ])
            system_text = f"This is a multi-agent conversation with the following agents:\n{agents_desc}"
        else:
            system_text = "This is a multi-agent conversation."
        
        unified_conv.add_system_message(system_text)

        # Collect and sort all messages by timestamp
        all_messages = []
        for agent_id in agents_to_include:
            conv = self.conversations[agent_id]
            agent = self.agents[agent_id]
            for msg in conv.messages:
                all_messages.append((agent, msg))
        
        # Sort by creation timestamp
        all_messages.sort(key=lambda x: x[1].created_at)
        
        # Add messages to unified conversation
        for agent, msg in all_messages:
            # Skip system messages except the first one we added
            if msg.role == MessageRole.SYSTEM:
                continue
                
            # Create a copy of the message to modify
            unified_msg = deepcopy(msg)
            
            # Add agent name to text contents if requested
            if include_agent_names and msg.role in {MessageRole.USER, MessageRole.ASSISTANT}:
                for content in unified_msg.contents:
                    if content.type == ContentType.TEXT:
                        content.data = agent_name_format.format(
                            agent_name=agent.name,
                            content=content.data
                        )
            
            # Add agent metadata
            unified_msg.metadata.update({
                "agent_id": agent.id,
                "agent_name": agent.name,
                "agent_author": agent.author,
                "agent_version": agent.version
            })
            
            # Add the modified message to the unified conversation
            unified_conv.add_message(unified_msg)
        
        return unified_conv

    def send_to_openai_sync(self, client, agent_ids: Optional[Set[str]] = None, **kwargs) -> Dict[str, Any]:
        """Send the unified conversation to OpenAI synchronously."""
        unified_conv = self.create_unified_conversation(agent_ids=agent_ids)
        return unified_conv.send_to_openai_sync(client, **kwargs)

    async def send_to_openai_async(self, client, agent_ids: Optional[Set[str]] = None, **kwargs) -> Dict[str, Any]:
        """Send the unified conversation to OpenAI asynchronously."""
        unified_conv = self.create_unified_conversation(agent_ids=agent_ids)
        return await unified_conv.send_to_openai_async(client, **kwargs)

    def save_to_json(self, filepath: str) -> None:
        """Save the entire multi-agent conversation system to a JSON file."""
        data = {
            "agents": {id: agent.to_dict() for id, agent in self.agents.items()},
            "conversations": {id: {
                "messages": [msg.to_dict() for msg in conv.messages],
                "metadata": conv.metadata,
                "created_at": conv.created_at.isoformat()
            } for id, conv in self.conversations.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "MultiAgentConversationManager":
        """Load a multi-agent conversation system from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        manager = cls(
            agents={id: Agent.from_dict(agent_data) 
                   for id, agent_data in data["agents"].items()},
            conversations={id: Conversation(
                messages=[MultimodalMessage.from_dict(msg) for msg in conv_data["messages"]],
                metadata=conv_data["metadata"],
                created_at=datetime.fromisoformat(conv_data["created_at"])
            ) for id, conv_data in data["conversations"].items()},
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
        return manager