# extractors.py

def qa_extractor(prompt: str) -> tuple[str, str]:
    """
    Extracts the question and static suffix from a structured QA prompt.
    
    Handles common cases such as:
        "Question: <text> Answer Choices: <options>"
        or just "Question: <text>"
        or even raw text without labels.
    
    Args:
        prompt: The full input text from a QA-style dataset.
    
    Returns:
        tuple[str, str]: (question_text, static_suffix)
            - question_text: the main, variable question portion
            - static_suffix: any fixed section (e.g., "Answer Choices: ...")
    """
    if not isinstance(prompt, str):
        raise TypeError(f"Input prompt must be str, got {type(prompt).__name__}.")
    
    if not prompt.strip():
        raise ValueError("Input prompt cannot be empty or whitespace.")
    
    prompt = prompt.strip()
    question_text, static_suffix = prompt, ""

    # Split off suffix if present
    if "Answer Choices:" in prompt:
        parts = prompt.split("Answer Choices:", 1)
        question_text = parts[0].strip()
        static_suffix = ("Answer Choices:" + parts[1]).strip()

    # Remove leading label only if it exists
    if question_text.startswith("Question:"):
        question_text = question_text[len("Question:"):].strip()

    if not question_text:
        raise ValueError("Extracted question text is empty after removing markers.")
    
    return question_text, static_suffix
