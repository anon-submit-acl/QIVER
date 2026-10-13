#!/usr/bin/env python3
"""Render WVS chat prompts from stable, structured dataset records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any


FIELD_ALIASES = {
    "qid": "Q_id",
    "country": "Country",
    "gender": "Gender",
    "marital_status": "Marital Status",
    "has_children": "Has Children",
    "education_level": "Education Level",
    "occupation": "Occupation",
    "work_nature": "Work Nature",
    "religion": "Religion",
    "life_stage": "Life Stage",
    "income_bracket": "Income Bracket",
    "common_language": "Common Language",
    "category": "Category",
    "question": "Question",
    "options": "Options",
}

IDENTITY_FIELD_ORDER = (
    "Country",
    "Q266",
    "Q272",
    "Q289",
    "Q289CS9",
    "Q261",
    "Q275",
    "Q278",
    "Q265",
    "Q262",
    "Q277",
    "Q281",
    "Q279",
    "Q267",
    "Q268",
    "Q264",
    "Q274",
    "Q286",
    "Q283",
    "Q260",
    "Q287",
    "Q288",
    "Q276",
    "Q280",
)

IDENTITY_FIELD_LABELS = {
    "Country": "Country",
    "Q266": "Birth country",
    "Q272": "Home language",
    "Q289": "Religion broad",
    "Q289CS9": "Religion detailed",
    "Q261": "Birth decade",
    "Q275": "Education level",
    "Q278": "Father education level",
    "Q265": "Father immigrant status",
    "Q262": "Age bracket",
    "Q277": "Mother education level",
    "Q281": "Occupation group",
    "Q279": "Employment status",
    "Q267": "Mother birth country",
    "Q268": "Father birth country",
    "Q264": "Mother immigrant status",
    "Q274": "Children count",
    "Q286": "Family savings",
    "Q283": "Father occupation at 14",
    "Q260": "Gender",
    "Q287": "Subjective social class",
    "Q288": "Income bracket",
    "Q276": "Spouse education level",
    "Q280": "Spouse employment status",
}

LEGACY_IDENTITY_ITEMS = (
    ("Country", "country"),
    ("Life stage", "life_stage"),
    ("Gender", "gender"),
    ("Marital status", "marital_status"),
    ("Children", "has_children"),
    ("Education level", "education_level"),
    ("Occupation", "occupation"),
    ("Work nature", "work_nature"),
    ("Income bracket", "income_bracket"),
    ("Home language", "common_language"),
    ("Religion", "religion"),
)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load WVS prompt templates.") from exc
            value = yaml.safe_load(stream)
        elif path.suffix.lower() == ".json":
            value = json.load(stream)
        else:
            raise ValueError(f"Unsupported prompt config type: {path}")
    if not isinstance(value, dict):
        raise ValueError(f"Prompt config must contain a mapping: {path}")
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _as_text(value: Any, default: str = "Unknown") -> str:
    if _is_missing(value):
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(item, default="") for item in value)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    return str(value)


def _parse_information(value: Any) -> dict[str, Any]:
    if _is_missing(value) or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Information must decode to a JSON object.")
        return parsed
    raise TypeError(f"Unsupported Information value: {type(value).__name__}")


def _parse_optional_mapping(value: Any) -> dict[str, Any]:
    if _is_missing(value) or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _render_identity_items(
    source: Mapping[str, Any],
    context: Mapping[str, str],
    selected_fields: list[str] | None = None,
) -> str:
    identity_values = _parse_optional_mapping(source.get("identity_json"))
    if identity_values:
        allowed = set(selected_fields) if selected_fields else None
        ordered_features = [
            feature
            for feature in IDENTITY_FIELD_ORDER
            if feature in identity_values and (allowed is None or feature in allowed)
        ]
        ordered_features.extend(
            sorted(
                feature
                for feature in identity_values
                if feature not in set(ordered_features)
                and (allowed is None or feature in allowed)
            )
        )
        return "\n".join(
            f"* {IDENTITY_FIELD_LABELS.get(feature, feature)}: {_as_text(identity_values.get(feature))}"
            for feature in ordered_features
        )

    legacy = {
        label: alias for label, alias in LEGACY_IDENTITY_ITEMS
    }
    if selected_fields:
        selected = set(selected_fields)
        return "\n".join(
            f"* {label}: {context.get(alias, 'Unknown')}"
            for label, alias in LEGACY_IDENTITY_ITEMS
            if label in selected or alias in selected
        )
    return "\n".join(f"* {label}: {context.get(alias, 'Unknown')}" for label, alias in LEGACY_IDENTITY_ITEMS)


class WVSPromptRenderer:
    """Compile a YAML/JSON prompt template and render OpenAI-style messages."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = _load_mapping(self.config_path)
        self.name = str(self.config.get("name", self.config_path.stem))
        self.version = str(self.config.get("version", "1"))
        self.system_template = Template(str(self.config.get("system_prompt", "")).strip())
        self.identity_templates = self._load_identity_templates()
        self.identity_template_ids = tuple(self.identity_templates)
        self.identity_template = self.identity_templates[self.identity_template_ids[0]]
        user_prompt = self.config.get("user_prompt")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError(f"Prompt config requires a non-empty user_prompt: {self.config_path}")
        self.user_template = Template(user_prompt.strip())

        output = self.config.get("output", {})
        if not isinstance(output, dict):
            raise ValueError("Prompt config output must be a mapping.")
        self.answer_start = str(output.get("answer_start", "<answer>"))
        self.answer_end = str(output.get("answer_end", "</answer>"))
        self.template_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()

    def _load_identity_templates(self) -> dict[str, Template]:
        variants = self.config.get("identity_templates")
        if variants is None:
            return {"default": Template(str(self.config.get("identity_template", "")).strip())}
        if not isinstance(variants, Mapping) or not variants:
            raise ValueError("Prompt config identity_templates must be a non-empty mapping.")

        templates: dict[str, Template] = {}
        for template_id, template_text in variants.items():
            normalized_id = str(template_id).strip()
            if not normalized_id:
                raise ValueError("Identity template IDs must be non-empty strings.")
            if not isinstance(template_text, str) or not template_text.strip():
                raise ValueError(f"Identity template {normalized_id!r} must contain non-empty text.")
            if normalized_id in templates:
                raise ValueError(f"Duplicate identity template ID: {normalized_id}")
            templates[normalized_id] = Template(template_text.strip())
        return templates

    def build_context(
        self,
        record: Mapping[str, Any],
        identity_template_id: str | None = None,
        extra_context: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        source = dict(record)
        information = _parse_information(source.get("Information"))
        context: dict[str, str] = {}

        for alias, source_key in FIELD_ALIASES.items():
            value = source.get(source_key)
            if _is_missing(value) or value == "":
                value = information.get(source_key)
            default = "" if alias in {"qid", "category"} else "Unknown"
            context[alias] = _as_text(value, default=default)
        selected_identity_fields: list[str] | None = None
        if extra_context and extra_context.get("identity_fields"):
            selected_identity_fields = [
                field.strip()
                for field in str(extra_context["identity_fields"]).split(",")
                if field.strip()
            ]
        context["identity_items"] = _render_identity_items(
            source, context, selected_fields=selected_identity_fields
        )

        selected_id = identity_template_id or self.identity_template_ids[0]
        try:
            selected_template = self.identity_templates[selected_id]
        except KeyError as exc:
            available = ", ".join(self.identity_template_ids)
            raise KeyError(f"Unknown identity template {selected_id!r}; available: {available}") from exc

        context["identity_profile"] = selected_template.substitute(context)
        context["identity_template_id"] = selected_id
        context["template_name"] = self.name
        context["template_version"] = self.version
        if extra_context:
            context.update({str(key): str(value) for key, value in extra_context.items()})
        context.setdefault("identity_guidance", "")
        return context

    def render_messages(
        self,
        record: Mapping[str, Any],
        identity_template_id: str | None = None,
        extra_context: Mapping[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        context = self.build_context(
            record, identity_template_id=identity_template_id, extra_context=extra_context
        )
        messages: list[dict[str, str]] = []
        system_prompt = self.system_template.substitute(context)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.user_template.substitute(context)})
        return messages

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "sha256": self.template_hash,
            "path": str(self.config_path),
            "identity_template_ids": list(self.identity_template_ids),
        }
