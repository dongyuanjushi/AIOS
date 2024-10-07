

from cerebrum.manager.manager import AgentManager


manager = AgentManager('https://my.aios.foundation/')
agent = manager.download_agent('example', 'academic_agent', '0.0.2')
print(agent)
agent = manager.load_agent('example', 'academic_agent', '0.0.1')
print(agent)