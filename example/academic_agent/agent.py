from cerebrum.agents.react import ReactAgent
import os


class AcademicAgent(ReactAgent):
    def __init__(self, agent_name, task_input, config):
        ReactAgent.__init__(
            self, agent_name, task_input, config
        )
        self.workflow_mode = "manual"
        # self.workflow_mode = "automatic"

    def check_path(self, tool_calls):
        script_path = os.path.abspath(__file__)
        save_dir = os.path.join(
            os.path.dirname(script_path), "output"
        )  # modify the customized output path for saving outputs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for tool_call in tool_calls:
            try:
                for k in tool_call["parameters"]:
                    if "path" in k:
                        path = tool_call["parameters"][k]
                        if not path.startswith(save_dir):
                            tool_call["parameters"][k] = os.path.join(
                                save_dir, os.path.basename(path)
                            )
            except Exception:
                continue
        return tool_calls

    def manual_workflow(self):
        workflow = [
            # {"message": "Gather research topic and keywords", "tool_use": []},
            {"message": "Search for relevant papers on arXiv", "tool_use": ["arxiv"]},
            {"message": "Provide responses based on the user's query", "tool_use": []},
            # {
            #     "message": "Identify research gaps and generate potential research questions",
            #     "tool_use": [],
            # },
        ]
        return workflow

    def run(self):
        return super().run()