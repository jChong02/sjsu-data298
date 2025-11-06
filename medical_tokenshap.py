from TokenSHAP import TokenSHAP
import re

def extract_sections(prompt):
    if "Answer Choices:" in prompt:
        parts = prompt.split("Answer Choices:", 1)
        question_text = parts[0].replace("Question:", "").strip()
        static_suffix = "Answer Choices:" + parts[1]
    elif "Respond in this format:" in prompt:
        parts = prompt.split("Respond in this format:", 1)
        question_text = parts[0].replace("Question:", "").strip()
        static_suffix = "Respond in this format:" + parts[1]
    else:
        question_text = prompt.strip()
        static_suffix = ""
    return question_text, static_suffix


class MedicalTokenSHAP(TokenSHAP):
    """
    TokenSHAP subclass that perturbs only the question part of
    medical QA prompts while keeping the suffix (Answer Choices)
    fixed.
    """

    def _get_samples(self, content: str):
        question_text, self.static_suffix = extract_sections(content)
        return self.splitter.split(question_text)

    def _prepare_combination_args(self, combination, original_content):
        # Reattach suffix before sending to model
        prompt = self.splitter.join(combination)
        if getattr(self, "static_suffix", ""):
            prompt = f"{prompt}\n\n{self.static_suffix}"
        return {"prompt": prompt}
