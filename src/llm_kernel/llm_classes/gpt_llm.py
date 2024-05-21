import re
from .base_llm import BaseLLMKernel
import time
from openai import OpenAI

from ...utils.message import Response

class GPTLLM(BaseLLMKernel):

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 256,
                 log_mode: str = "console"):
        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode)

    def load_llm_and_tokenizer(self) -> None:
        self.model = OpenAI()
        self.tokenizer = None

    def process(self,
            llm_request,
            temperature=0.0
        ):
        assert re.search(r'gpt', self.model_name, re.IGNORECASE)
        # agent_process.set_status("executing")
        # agent_process.set_start_time(time.time())
        prompt = llm_request.message.prompt
        # self.logger.log(
        #     f"{agent_process.agent_name} is switched to executing.\n",
        #     level = "executing"
        # )
        time.sleep(2)
        result = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools = agent_process.message.tools,
            tool_choice = "required" if agent_process.message.tools else None
        )

        # print(response.choices[0].message)
        agent_process.set_response(
            Response(
                response_message = response.choices[0].message.content,
                tool_calls = response.choices[0].message.tool_calls
            )
        )
        agent_process.set_status("done")
        agent_process.set_end_time(time.time())
        return
