from cerebrum.llm.base import BaseLLM
from cerebrum.utils.chat import Query


class AgentProcessor:
    @staticmethod
    def process_response(query: Query, llm: BaseLLM):
        print(query)
        print(llm)
        return llm.execute(query)