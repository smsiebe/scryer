"""compiler.py - Compiles programmatic (Pydantic) and declarative (YAML/JSON) templates into JSON Schemas and prompt hints."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import jsonschema
from pydantic import BaseModel, ValidationError
import yaml

from scryer.contracts import ExtractionTemplate
from scryer.protocols import TemplateEngineProtocol


def inline_json_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve and inline local ``#/$defs/...`` and ``#/definitions/...`` refs.

    Many local LLM/VLM JSON-schema constrained decoders reject schemas that contain
    ``$defs`` or ``$ref`` pointers. Cycle detection prevents infinite recursion on
    self-referential schemas.
    """
    root = copy.deepcopy(schema)
    defs: Dict[str, Any] = {}
    if isinstance(root.get("$defs"), dict):
        defs.update(root.pop("$defs"))
    if isinstance(root.get("definitions"), dict):
        defs.update(root.pop("definitions"))

    def _resolve(node: Any, stack: Tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [_resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            parts = ref[2:].split("/")
            target_name = parts[-1] if len(parts) >= 2 and parts[0] in ("$defs", "definitions") else None
            if target_name and target_name in defs and target_name not in stack:
                resolved = _resolve(copy.deepcopy(defs[target_name]), stack + (target_name,))
                # Preserve sibling metadata (e.g. description, default) alongside resolved target
                siblings = {k: v for k, v in node.items() if k != "$ref"}
                if isinstance(resolved, dict) and siblings:
                    merged = dict(resolved)
                    merged.update({k: _resolve(v, stack) for k, v in siblings.items()})
                    return merged
                return resolved

        return {
            k: _resolve(v, stack)
            for k, v in node.items()
            if k not in ("$defs", "definitions")
        }

    return _resolve(root, ())


def validate_structured_output(
    data: Any,
    schema: Optional[Dict[str, Any]] = None,
    model_cls: Optional[Type[BaseModel]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate extracted Phase 2 JSON payload against Pydantic model and/or JSON Schema.

    Returns ``(validated_dict, validation_annotations)`` preserving raw fields
    in annotations on partial conformance so no extracted context is lost.
    """
    annotations: List[str] = []
    if not isinstance(data, dict):
        if data is None:
            return {}, annotations
        return {}, [f"Structured output expected JSON object, got {type(data).__name__}."]

    normalized: Dict[str, Any] = dict(data)

    if model_cls is not None:
        try:
            validated = model_cls.model_validate(data)
            normalized = validated.model_dump(mode="json")
            # Capture unrecognized keys in annotations so nothing is dropped
            extra_keys = sorted(set(data.keys()) - set(model_cls.model_fields.keys()))
            if extra_keys:
                annotations.append(
                    f"Additional unmodeled fields preserved: {json.dumps({k: data[k] for k in extra_keys})}"
                )
            return normalized, annotations
        except ValidationError as exc:
            annotations.append(f"Pydantic schema validation notice: {exc.errors(include_url=False)}")

    if schema:
        try:
            jsonschema.validate(instance=normalized, schema=schema)
        except jsonschema.ValidationError as exc:
            annotations.append(f"JSON Schema validation notice at {list(exc.path)}: {exc.message}")
        except jsonschema.SchemaError as exc:
            annotations.append(f"Invalid target schema notice: {exc.message}")

    return normalized, annotations

DEFAULT_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Concise summary of the document, scene, or image content.",
        },
        "extracted_text": {
            "type": "string",
            "description": "Visible text, signs, labels, or tabular data.",
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "key_values": {
            "type": "object",
            "description": "Key-value pairs extracted from forms, headers, or metadata.",
        },
    },
    "required": ["summary"],
}


class TemplateCompiler(TemplateEngineProtocol):
    """Compiles diverse template specifications into standard JSON Schemas and prompt guidelines."""

    def resolve_template(
        self, template: Any
    ) -> Tuple[Dict[str, Any], List[str], Optional[Type[BaseModel]]]:
        """Compile template and return ``(inlined_schema, guidelines, optional_pydantic_cls)``."""
        if template is None:
            return (
                inline_json_schema_refs(DEFAULT_EXTRACTION_SCHEMA),
                [
                    "Extract a concise summary of the document/image.",
                    "Capture all visible text, tables, and entities.",
                ],
                None,
            )

        # 1. Pydantic BaseModel subclass
        if isinstance(template, type) and issubclass(template, BaseModel):
            schema = inline_json_schema_refs(template.model_json_schema())
            guidelines = self._extract_pydantic_guidelines(template)
            return schema, guidelines, template

        # 2. ExtractionTemplate contract instance
        if isinstance(template, ExtractionTemplate):
            schema = inline_json_schema_refs(template.to_response_schema())
            return schema, list(template.prompt_guidelines), None

        # 3. Named preset or path to file (YAML or JSON)
        if isinstance(template, str):
            norm = template.strip().lower().replace("-", "_").replace(" ", "_")
            from scryer.templates.presets import (
                INVOICE_TEMPLATE,
                SCENE_TEMPLATE,
                SUMMARY_TEMPLATE,
            )

            preset_map = {
                "invoice": INVOICE_TEMPLATE,
                "invoice_extraction": INVOICE_TEMPLATE,
                "scene": SCENE_TEMPLATE,
                "scene_analysis": SCENE_TEMPLATE,
                "visual_scene": SCENE_TEMPLATE,
                "visual_scene_extraction": SCENE_TEMPLATE,
                "summary": SUMMARY_TEMPLATE,
                "document_summary": SUMMARY_TEMPLATE,
            }
            if norm in preset_map:
                return self.resolve_template(preset_map[norm])

        if isinstance(template, (str, Path)):
            path_obj = Path(template)
            if path_obj.exists() and path_obj.is_file():
                raw = path_obj.read_text(encoding="utf-8")
                if path_obj.suffix.lower() in (".yaml", ".yml"):
                    parsed = yaml.safe_load(raw)
                else:
                    parsed = json.loads(raw)
                schema, guidelines = self._compile_dict_template(parsed)
                return inline_json_schema_refs(schema), guidelines, None
            raise FileNotFoundError(f"Template preset or file not found: {template}")

        # 4. Dictionary specification
        if isinstance(template, dict):
            schema, guidelines = self._compile_dict_template(template)
            return inline_json_schema_refs(schema), guidelines, None

        # Fallback to default
        return inline_json_schema_refs(DEFAULT_EXTRACTION_SCHEMA), [], None

    def compile(self, template: Any) -> Tuple[Dict[str, Any], List[str]]:
        """Compile a template specification into (response_schema_dict, prompt_guidelines).

        Supported inputs:
        - None: returns default generic extraction schema and prompt hints
        - Pydantic BaseModel subclass: generates schema from model_json_schema() and field descriptions
        - ExtractionTemplate instance: returns schema_dict and prompt_guidelines
        - dict: if contains 'fields' or 'properties', compiles to JSON Schema; else treats as schema dict
        - str or Path: path to YAML or JSON template file
        """
        schema, guidelines, _ = self.resolve_template(template)
        return schema, guidelines

    def _extract_pydantic_guidelines(self, model_cls: Type[BaseModel]) -> List[str]:
        guidelines = []
        if model_cls.__doc__:
            doc = model_cls.__doc__.strip()
            if doc:
                guidelines.append(doc)

        for name, field_info in model_cls.model_fields.items():
            desc = field_info.description or ""
            if desc:
                guidelines.append(f"Extract field '{name}': {desc}")
            else:
                guidelines.append(f"Extract field '{name}'")
        return guidelines

    def _compile_dict_template(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        # Check if already a pure JSON schema
        if "type" in data and ("properties" in data or "items" in data):
            guidelines = data.get("guidelines") or data.get("prompt_guidelines") or []
            return data, list(guidelines)

        # Declarative schema definition
        name = data.get("name", "CustomTemplate")
        desc = data.get("description", "")
        guidelines = list(data.get("guidelines") or data.get("prompt_guidelines") or [])
        if desc:
            guidelines.insert(0, desc)

        fields = data.get("fields", {})
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for field_name, field_def in fields.items():
            if isinstance(field_def, dict):
                prop_type = field_def.get("type", "string")
                prop_desc = field_def.get("description", "")
                prop_enum = field_def.get("enum")
                prop: Dict[str, Any] = {"type": prop_type}
                if prop_desc:
                    prop["description"] = prop_desc
                    guidelines.append(f"Field '{field_name}': {prop_desc}")
                if prop_enum:
                    prop["enum"] = prop_enum
                if field_def.get("required", False):
                    required.append(field_name)
                properties[field_name] = prop
            elif isinstance(field_def, str):
                properties[field_name] = {"type": field_def}
                guidelines.append(f"Field '{field_name}' ({field_def})")

        schema: Dict[str, Any] = {
            "title": name,
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema, guidelines

