

from cerebrum.manager.manager import AgentManager
from example.academic_agent.agent import AcademicAgent
from cerebrum.llm.adapter import LLMAdapter
from cerebrum.runtime.process import AgentProcessor
from cerebrum.interface import AutoAgentGenerator


# a = LLMAdapter('gpt-4o')
# academic_agent = AcademicAgent(
#     'agent1',
#     'tell me about turtles',
#     {
#     "name": "academic_agent",
#     "description": [
#         "You are an academic research assistant. ",
#         "Help users find relevant research papers, summarize key findings, and generate potential research questions."
#     ],
#     "tools": [
#         "arxiv/arxiv"
#     ],
#     "meta": {
#         "author": "example",
#         "version": "0.0.3",
#         "license": "CC0"
#     },
#     "build": {
#         "entry": "agent.py",
#         "module": "AcademicAgent"
#     }
# }
# )

# academic_agent.llm = a.get_model()

# res = academic_agent.run()
# print(res)

# manager = AgentManager('https://my.aios.foundation/')
# agent_data = manager.package_agent('/Users/rama2r/AIOS/example/academic_agent')

# manager.upload_agent(agent_data)
# manager.download_agent('example', 'academic_agent', '0.0.3')
# print(agent)
# agent = manager.load_agent('example', 'academic_agent', '0.0.1')
# print(agent)

# agent = AutoAgentGenerator.build_agent('example/academic_agent', 'gpt-4o')
# res = agent.run('tell me about fish')
# print(res)

from cerebrum.interface import AutoAgent, AutoLLM
from cerebrum.runtime import Pipeline

academic_agent, academic_agent_config = AutoAgent.from_pretrained('example/academic_agent')
gpt_llm = AutoLLM.from_foundational('gpt-4o')
pipeline = Pipeline()

pipeline \
  .add_agent(academic_agent, academic_agent_config, 0) \
  .add_llm(gpt_llm, 0)

res = pipeline.run('tell me about fish')
print(res)