"""The exact text each checkpoint sees, and the clean-up of what it says.

Used both for generation and for activation extraction, so the two can
never disagree about the context.
"""
from overrefusal import config


SYSTEM_TURN = "<|im_start|>system"
USER_TURN = "<|im_start|>user"


def build_prompt(tok, user_message: str, stage: str, variant: str = "none") -> str:
    """Base model: the minimal frame in config.BASE_FRAME. Others: their chat
    template with no system prompt passed.

    variant "none": the template as is; OLMo 3's inserts its default system
    prompt. variant "nosys": that system turn is removed, so the text starts
    at the user turn (no change for templates without one, like OLMo 2's).
    An empty system message would not do: the template still writes the turn.
    """
    if stage == "base":
        return config.BASE_FRAME.format(message=user_message)
    text = tok.apply_chat_template([{"role": "user", "content": user_message}],
                                   tokenize=False, add_generation_prompt=True)
    if variant == "none":
        return text
    if variant != "nosys":
        raise ValueError(f"unknown variant {variant!r}")
    if SYSTEM_TURN in text:
        start = text.find(USER_TURN)
        if start < 0 or text.find(SYSTEM_TURN) > start:
            raise ValueError("cannot locate the system turn before the user turn")
        text = text[start:]
    if SYSTEM_TURN in text:
        raise ValueError("system turn still present after removal")
    return text


def clean_response(text: str, stage: str) -> str:
    """Drop the dialogue turns the base model invents after its answer."""
    return text.split(config.BASE_STOP, 1)[0] if stage == "base" else text
