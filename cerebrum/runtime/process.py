from typing import Any, Type

# from cerebrum.agents.base import BaseAgent
# from cerebrum.llm.base import BaseLLM
from cerebrum.utils.chat import Query

class RunnableAgent:
    def __init__(self, agent_class, config, llm):
        self.agent = agent_class
        self.config = config
        self.llm = llm

    def run(self, query):
        _runnable = AgentProcessor.make_runnable(
            self.agent,
            query,
            self.config
        )

        _runnable.llm = self.llm

        return _runnable.run()

class AgentProcessor:
    @staticmethod
    def process_response(query: Query, llm: Any):
        print(query)
        print(llm)
        return llm.execute(query)

    @staticmethod
    def make_runnable(agent_class: Type[Any], query: str, config: dict):
        _agent = agent_class(
            'test',
            query,
            config
        )

        return _agent

