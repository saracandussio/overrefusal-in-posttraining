# tests/test_olmo_loader.py
from unittest.mock import MagicMock, patch
from models.olmo_loader import CheckpointModel, iter_checkpoints


def make_model(checkpoint_name, has_chat_template=True, system_prompt=None):
    with patch.object(CheckpointModel, "__init__", lambda self, **kw: None):
        m = CheckpointModel.__new__(CheckpointModel)
        m.checkpoint_name = checkpoint_name
        m.system_prompt = system_prompt
        m.tokenizer = MagicMock()
        m.tokenizer.chat_template = "dummy" if has_chat_template else None
        m.tokenizer.apply_chat_template = lambda msgs, **kw: f"[TEMPLATE]{msgs[-1]['content']}[/TEMPLATE]"
        return m


# --- _build_prompt ---

def test_base_returns_raw():
    m = make_model("base__none")
    assert m._build_prompt("hello") == "hello"

def test_base_ignores_system_prompt():
    m = make_model("base__none", system_prompt="Be safe.")
    assert m._build_prompt("hello") == "hello"

def test_finetuned_uses_chat_template():
    m = make_model("sft__none")
    result = m._build_prompt("hello")
    assert "[TEMPLATE]" in result and "hello" in result

def test_finetuned_fallback_no_template():
    m = make_model("sft__none", has_chat_template=False, system_prompt="Be safe.")
    assert m._build_prompt("hello") == "hello"

def test_finetuned_fallback_no_system_prompt():
    m = make_model("sft__none", has_chat_template=False)
    assert m._build_prompt("hello") == "hello"


# --- iter_checkpoints ---

def test_iter_skips_base_mistral_safety():
    checkpoint_map = {"base": "allenai/OLMo-base", "sft": "allenai/OLMo-sft"}
    system_prompts = {"none": None, "mistral_safety": "Be safe."}

    with patch("models.olmo_loader.CheckpointModel") as MockCM:
        MockCM.side_effect = lambda **kw: MagicMock(checkpoint_name=kw["checkpoint_name"])
        tags = [m.checkpoint_name for m in iter_checkpoints(checkpoint_map, system_prompts)]

    assert "base__mistral_safety" not in tags
    assert "base__none" in tags
    assert "sft__none" in tags
    assert "sft__mistral_safety" in tags


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
