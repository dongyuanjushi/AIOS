import re

from .base_llm import BaseLLMKernel
import time
from ...utils.utils import get_from_env

from ...utils.message import Response
class GeminiLLM(BaseLLMKernel):
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
        assert self.model_name == "gemini-pro"
        try:
            import google.generativeai as genai
            gemini_api_key = get_from_env("GEMINI_API_KEY")
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.tokenizer = None
        except ImportError:
            raise ImportError(
                "Could not import google.generativeai python package. "
                "Please install it with `pip install google-generativeai`."
            )

    def process(self,
                llm_request,
                temperature=0.0) -> None:
        assert re.search(r'gemini', self.model_name, re.IGNORECASE)
        prompt = llm_request.message.prompt
        # TODO: add tool calling

        time.sleep(2)
        outputs = self.model.generate_content(
            prompt
        )
        try:
            result = outputs.candidates[0].content.parts[0].text
        except IndexError:
            raise IndexError(f"{self.model_name} can not generate a valid result, please try again")

        llm_request.set_status("done")
        return Response(
            response_message = result
        )
