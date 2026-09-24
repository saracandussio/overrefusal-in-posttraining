"""The exact text each checkpoint sees, and the clean-up of what it says.

Used both for generation and for activation extraction, so the two can
never disagree about the context.
"""
from overrefusal import config


def build_prompt(tok, user_message: str, stage: str) -> str:
    """Base model: the minimal frame in config.BASE_FRAME. Others: their chat
    template, no system prompt."""
    if stage == "base":
        return config.BASE_FRAME.format(message=user_message)
    return tok.apply_chat_template([{"role": "user", "content": user_message}],
                                   tokenize=False, add_generation_prompt=True)


def clean_response(text: str, stage: str) -> str:
    """Drop the dialogue turns the base model invents after its answer."""
    return text.split(config.BASE_STOP, 1)[0] if stage == "base" else text
