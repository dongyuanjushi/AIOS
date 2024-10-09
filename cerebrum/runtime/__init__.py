
from cerebrum.agents.base import BaseAgent
from cerebrum.llm.base import BaseLLM


class Pipeline:
    def __init__(self):
        self.agents = dict()
        self.llms = dict()


    def add_agent(self, agent: BaseAgent, order: int):
        self.agents[order] = agent
        return self

    def add_llm(self, llm: BaseLLM, order: int):
        self.llms[order] = llm
        return self

    
    def run(self, query: str):
        agent_keys, llm_keys = list(self.agents.keys()), list(self.llms.keys())

        if len(agent_keys) != len (llm_keys):
            return False
        
        agent_keys.sort()

        for k in agent_keys:
            # support single step pipelines for now
            if k != agent_keys[-1]:
                return False
            else:
                self.agents[k].llm = self.llms[k]
