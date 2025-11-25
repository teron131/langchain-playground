import re
from typing import Any, Optional, Union

from pydantic import BaseModel


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and normalizing.

    Args:
        text: The text to clean

    Returns:
        Cleaned text string
    """
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", r"\n\n", text)
    # Remove excessive spaces
    text = re.sub(r" {2,}", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def clean_youtube_url(url: str) -> str:
    """Clean and normalize a YouTube URL.

    Args:
        url: YouTube URL in various formats

    Returns:
        Cleaned YouTube URL
    """
    # Remove query parameters except v
    if "youtube.com/watch" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    elif "youtu.be/" in url:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube URL.

    Args:
        url: URL to check

    Returns:
        True if URL is a YouTube URL, False otherwise
    """
    youtube_patterns = [
        r"youtube\.com/watch\?v=",
        r"youtu\.be/",
        r"youtube\.com/embed/",
        r"youtube\.com/v/",
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def schema_to_string(schema: Union[dict[str, Any], BaseModel]) -> str:
    """Parse a Pydantic BaseModel or a JSON schema and return a string representation of the schema.

    This provides context to LLM and avoids JSON format causing LangChain errors.

    Args:
        schema: Pydantic BaseModel class or JSON schema dict

    Returns:
        String representation of the schema
    """
    def _parse_properties(
        properties: dict[str, Any],
        required_fields: list[str],
        defs: dict[str, Any],
    ) -> tuple[list[str], set[str]]:
        lines = []
        refs = set()

        for name, spec in properties.items():
            type_str, type_refs = _type_string(spec, defs)
            refs |= type_refs
            desc = spec.get("description")
            lines.append(f"{name}: {type_str}" + (f" = {desc}" if desc else ""))

        return lines, refs

    def _type_string(spec: dict[str, Any], defs: dict[str, Any]) -> tuple[str, set[str]]:
        # $ref
        if "$ref" in spec:
            ref = spec["$ref"]
            if ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                return name, {name}

        # anyOf
        if "anyOf" in spec:
            types = []
            refs = set()
            for opt in spec["anyOf"]:
                if opt.get("type") == "null":
                    continue
                type_str, type_refs = _type_string(opt, defs)
                types.append(type_str)
                refs |= type_refs
            return (" | ".join(sorted(set(types))) if types else "Any"), refs

        # arrays
        if spec.get("type") == "array":
            item = spec.get("items", {})
            type_str, type_refs = _type_string(item, defs)
            return f"list[{type_str}]", type_refs

        # simple types
        return _simple_type(spec), set()

    def _simple_type(spec: dict[str, Any]) -> str:
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "object": "dict",
            "array": "list[Any]",
        }
        return type_mapping.get(spec.get("type", "Any"), spec.get("type", "Any"))

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    elif isinstance(schema, BaseModel):
        schema = schema.model_json_schema()

    lines = []
    defs = schema.get("$defs", {})

    main_lines, queued = _parse_properties(schema.get("properties", {}), schema.get("required", []), defs)
    lines.extend(main_lines)

    seen = set()
    while queued:
        name = queued.pop()
        if name in seen or name not in defs or defs[name].get("type") != "object":
            continue
        seen.add(name)

        lines.extend(["", f"## {name} Type", ""])
        def_lines, new_refs = _parse_properties(defs[name].get("properties", {}), defs[name].get("required", []), defs)
        lines.extend(f"  {line}" for line in def_lines)
        queued |= new_refs - seen

    return "\n".join(lines)
