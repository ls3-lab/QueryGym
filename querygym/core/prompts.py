from __future__ import annotations
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Union

_ROLES = ("system", "user", "assistant")


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""


def _construct_unique_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        # Merge keys (<<) are expanded, and unhashable keys reported, by construct_mapping
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError:
            continue
        if duplicate:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass
class PromptSpec:
    """Specification for a single prompt template.

    Attributes:
        id: Unique prompt identifier
        method_family: Method family this prompt belongs to
        version: Prompt version number
        template: Template dict with 'system', 'user', 'assistant' keys, or a list of
            {'role', 'content'} messages in conversation order
        meta: Additional metadata (authors, license, etc.)
    """

    id: str
    method_family: str
    version: int
    template: Union[Dict[str, str], List[Dict[str, str]]]
    meta: Dict[str, Any]


class PromptBank:
    """Manages prompt templates from a YAML file.

    Loads and provides access to prompt templates with metadata.
    Supports rendering templates with variable substitution.

    Attributes:
        _by_id: Internal dict mapping prompt IDs to PromptSpec objects

    Example:
        >>> from pathlib import Path
        >>> pb = PromptBank(Path("querygym/prompt_bank.yaml"))
        >>> messages = pb.render("genqr_keywords", query="diabetes")
        >>> print(messages)
    """

    def __init__(self, path: str | Path):
        """Initialize the prompt bank from a YAML file.

        Args:
            path: Path to the prompt bank YAML file

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML is malformed or a mapping repeats a key
            ValueError: If a message in a list template lacks a valid role or content
        """
        items = yaml.load(Path(path).read_text(), Loader=_UniqueKeyLoader) or []
        self._by_id: Dict[str, PromptSpec] = {}
        for x in items:
            if isinstance(x["template"], list):
                for i, m in enumerate(x["template"]):
                    if not (
                        isinstance(m, dict)
                        and m.get("role") in _ROLES
                        and isinstance(m.get("content"), str)
                    ):
                        raise ValueError(
                            f"Prompt {x['id']!r}: template message {i} needs a role "
                            f"({', '.join(_ROLES)}) and a string content"
                        )
            self._by_id[x["id"]] = PromptSpec(
                id=x["id"],
                method_family=x.get("method_family", ""),
                version=x.get("version", 1),
                template=x["template"],
                meta={
                    k: v
                    for k, v in x.items()
                    if k not in ["id", "method_family", "version", "template"]
                },
            )

    def render(self, prompt_id: str, **vars) -> List[Dict[str, str]]:
        """Render a prompt template with variable substitution.

        A list template is rendered message by message in its order. A dict template
        is rendered as system, user, then assistant, each only if present.

        Args:
            prompt_id: ID of the prompt to render
            **vars: Variables to substitute in the template (e.g., query="text")

        Returns:
            List of message dicts with 'role' and 'content' keys,
            ready for LLM chat completion

        Raises:
            KeyError: If prompt_id doesn't exist
            KeyError: If required template variables are missing

        Example:
            >>> messages = pb.render("genqr_keywords", query="diabetes")
            >>> # Returns: [{"role": "system", "content": "..."}, ...]
        """
        spec = self._by_id[prompt_id]

        # Ordered messages (multi-turn, e.g. few-shot examples)
        if isinstance(spec.template, list):
            return [
                {"role": m["role"], "content": m["content"].format(**vars)}
                for m in spec.template
            ]

        messages = []

        # System message (if present)
        if "system" in spec.template:
            sys = spec.template.get("system", "").format(**vars)
            messages.append({"role": "system", "content": sys})

        # User message (if present)
        if "user" in spec.template:
            usr = spec.template.get("user", "").format(**vars)
            messages.append({"role": "user", "content": usr})

        # Assistant message (if present) - for priming responses
        if "assistant" in spec.template:
            asst = spec.template.get("assistant", "").format(**vars)
            messages.append({"role": "assistant", "content": asst})

        return messages

    def list(self) -> List[str]:
        """List all available prompt IDs.

        Returns:
            List of prompt IDs
        """
        return list(self._by_id.keys())

    def get(self, prompt_id: str) -> PromptSpec:
        """Get a prompt specification by ID.

        Args:
            prompt_id: ID of the prompt

        Returns:
            PromptSpec object

        Raises:
            KeyError: If prompt_id doesn't exist
        """
        return self._by_id[prompt_id]

    def get_meta(self, prompt_id: str) -> Dict[str, Any]:
        """Get metadata for a prompt.

        Args:
            prompt_id: ID of the prompt

        Returns:
            Metadata dict (authors, license, etc.)

        Raises:
            KeyError: If prompt_id doesn't exist
        """
        return self._by_id[prompt_id].meta
