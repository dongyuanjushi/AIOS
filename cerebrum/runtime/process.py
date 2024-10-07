from cerebrum.llm.base import BaseLLM
from cerebrum.utils.chat import Query


class AgentProcessor:
    @classmethod
    def process_response(query: Query, llm: BaseLLM):
        return llm.execute(query)