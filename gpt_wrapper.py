"""
gpt_wrapper.py

Lightweight wrapper for OpenAI GPT models (e.g., gpt-5-nano) to match the
interface expected by TokenSHAP and other explainability tools.

This class mimics the structure of MedGemmaQAWrapper but uses the OpenAI
Responses API for fast inference. Designed for quick local testing.
"""

import openai


class GPTQAWrapper:
    """
    Simple OpenAI GPT wrapper that exposes a `.generate(prompt)` method
    compatible with TokenSHAP and other explainability pipelines.

    Attributes:
        model_name (str): The OpenAI model identifier (e.g. 'gpt-5-nano')
        client (openai.OpenAI): OpenAI API client
        task_type (str): 'yn', 'mcq', or 'free' (for consistency with other wrappers)
    """

    def __init__(self, model_name="gpt-5-nano", api_key=None):
        """
        Initialize GPT model wrapper.

        Args:
            model_name: Name of the OpenAI model (default: 'gpt-5-nano')
            api_key: Optional API key. If not provided, uses environment variable OPENAI_API_KEY
        """
        api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.task_type = "free"

    def set_task(self, task_type):
        """
        Set the task type ('yn', 'mcq', or 'free').
        Included for interface consistency with MedGemmaQAWrapper.
        """
        self.task_type = task_type

    def generate(self, prompt):
        """
        Generate a structured response to a prompt.
        The OpenAI API handles all decoding internally.

        Returns:
            str: Model's raw output text.
        """
        if self.task_type in {"yn", "mcq"}:
            prompt = (
                f"{prompt.strip()}\n\n"
                "Respond in this format:\n"
                "Answer: <answer label only>\n"
                "Rationale: <brief justification>"
                )
        else:
            prompt = (
            f"{prompt.strip()}\n\n"
            "Respond in this format:\n"
            "Answer: <short text answer>"
            )
        
        response = self.client.responses.create(model=self.model_name,input=prompt)

        raw_response = response.output_text.strip()
        return raw_response
