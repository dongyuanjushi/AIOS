

from cerebrum.manager.manager import AgentManager
from example.academic_agent.agent import AcademicAgent
from cerebrum.llm.adapter import LLMAdapter
from cerebrum.runtime.process import AgentProcessor

a = LLMAdapter('gpt-4o')
academic_agent = AcademicAgent(
    'agent1',
    'tell me about turtles',
    {
    "name": "academic_agent",
    "description": [
        "You are an academic research assistant. ",
        "Help users find relevant research papers, summarize key findings, and generate potential research questions."
    ],
    "tools": [
        "arxiv/arxiv"
    ],
    "meta": {
        "author": "example",
        "version": "0.0.3",
        "license": "CC0"
    },
    "build": {
        "entry": "agent.py",
        "module": "AcademicAgent"
    }
}
)

academic_agent.llm = a.get_model()

res = academic_agent.run()
print(res)

# manager = AgentManager('https://my.aios.foundation/')
# agent = manager.download_agent('example', 'academic_agent', '0.0.2')
# print(agent)
# agent = manager.load_agent('example', 'academic_agent', '0.0.1')
# print(agent)