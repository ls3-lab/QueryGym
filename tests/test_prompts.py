from querygym.core.prompts import PromptBank
from pathlib import Path

import pytest
import yaml

BANK = Path(__file__).parents[1] / "querygym" / "prompt_bank.yaml"


def write_bank(tmp_path, text):
    path = tmp_path / "bank.yaml"
    path.write_text(text)
    return path


def test_prompt_bank_loads():
    pb = PromptBank(BANK)
    assert len(pb.list()) > 0


def test_list_template_renders_in_order(tmp_path):
    path = write_bank(tmp_path, """
- id: t.v1
  template:
    - role: system
      content: sys
    - role: user
      content: "example"
    - role: assistant
      content: "answer"
    - role: user
      content: "{query}"
""")
    messages = PromptBank(path).render("t.v1", query="q")
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "example"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "q"},
    ]


def test_duplicate_key_is_rejected(tmp_path):
    path = write_bank(tmp_path, """
- id: t.v1
  template:
    user: first
    user: second
""")
    with pytest.raises(yaml.YAMLError, match="duplicate key 'user'"):
        PromptBank(path)


def test_merge_keys_still_load(tmp_path):
    path = write_bank(tmp_path, """
- id: base.v1
  template: &base
    system: sys
    user: "{query}"
- id: t.v1
  template:
    <<: *base
    user: "Q: {query}"
""")
    messages = PromptBank(path).render("t.v1", query="q")
    assert messages == [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q: q"}]


def test_unhashable_key_is_a_yaml_error(tmp_path):
    path = write_bank(tmp_path, "- id: t.v1\n  template:\n    [1, 2]: x\n")
    with pytest.raises(yaml.YAMLError, match="unhashable key"):
        PromptBank(path)


@pytest.mark.parametrize("message",["{role: tool, content: x}", "{role: user}", "just text"])
def test_list_template_message_is_validated(tmp_path, message):
    path = write_bank(tmp_path, f"""
- id: t.v1
  template:
    - {message}
""")
    with pytest.raises(ValueError, match="template message 0"):
        PromptBank(path)


def test_csqe_sends_example_before_query():
    messages = PromptBank(BANK).render("csqe.v1", query="QUERY", contexts="CONTEXTS")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "how are some sharks warm blooded" in messages[1]["content"]
    assert "QUERY" in messages[3]["content"] and "CONTEXTS" in messages[3]["content"]
