"""compiler.py - Compiles programmatic (Pydantic) and declarative (YAML/JSON) templates into JSON Schemas and prompt hints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel
import yaml

from scryer.contracts import ExtractionTemplate
from scryer.protocols import TemplateEngineProtocol

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

    def compile(self, template: Any) -> Tuple[Dict[str, Any], List[str]]:
        """Compile a template specification into (response_schema_dict, prompt_guidelines).

        Supported inputs:
        - None: returns default generic extraction schema and prompt hints
        - Pydantic BaseModel subclass: generates schema from model_json_schema() and field descriptions
        - ExtractionTemplate instance: returns schema_dict and prompt_guidelines
        - dict: if contains 'fields' or 'properties', compiles to JSON Schema; else treats as schema dict
        - str or Path: path to YAML or JSON template file
        """
        if template is None:
            return DEFAULT_EXTRACTION_SCHEMA, [
                "Extract a concise summary of the document/image.",
                "Capture all visible text, tables, and entities.",
            ]

        # 1. Pydantic BaseModel subclass
        if isinstance(template, type) and issubclass(template, BaseModel):
            schema = template.model_json_schema()
            guidelines = self._extract_pydantic_guidelines(template)
            return schema, guidelines

        # 2. ExtractionTemplate contract instance
        if isinstance(template, ExtractionTemplate):
            schema = template.to_response_schema()
            return schema, template.prompt_guidelines

        # 3. Path to file (YAML or JSON)
        if isinstance(template, (str, Path)):
            path_obj = Path(template)
            if path_obj.exists() and path_obj.is_file():
                raw = path_obj.read_text(encoding="utf-8")
                if path_obj.suffix.lower() in (".yaml", ".yml"):
                    parsed = yaml.safe_load(raw)
                else:
                    parsed = json.loads(raw)
                return self._compile_dict_template(parsed)

        # 4. Dictionary specification
        if isinstance(template, dict):
            return self._compile_dict_template(template)

        # Fallback to default
        return DEFAULT_EXTRACTION_SCHEMA, []

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
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema, guidelines
