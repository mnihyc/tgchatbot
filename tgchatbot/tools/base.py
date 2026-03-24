from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tgchatbot.domain.models import ToolResult


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    user_display_name: str


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    transformed = _openai_strict_schema(schema)
    enum_values = transformed.get('enum')
    if isinstance(enum_values, list) and None not in enum_values:
        transformed['enum'] = [*enum_values, None]
    schema_type = transformed.get('type')
    if schema_type is None:
        transformed['type'] = ['null']
    elif isinstance(schema_type, list):
        if 'null' not in schema_type:
            transformed['type'] = [*schema_type, 'null']
    else:
        if schema_type != 'null':
            transformed['type'] = [schema_type, 'null']
    return transformed


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    transformed: dict[str, Any] = {}
    for key, value in schema.items():
        if key == 'properties' and isinstance(value, dict):
            transformed[key] = {name: _openai_strict_schema(prop) if isinstance(prop, dict) else prop for name, prop in value.items()}
        elif key == 'items' and isinstance(value, dict):
            transformed[key] = _openai_strict_schema(value)
        else:
            transformed[key] = value

    if transformed.get('type') == 'object' and isinstance(transformed.get('properties'), dict):
        original_required = set(transformed.get('required') or [])
        transformed['required'] = list(transformed['properties'].keys())
        for name, prop in list(transformed['properties'].items()):
            if not isinstance(prop, dict):
                continue
            if name not in original_required:
                transformed['properties'][name] = _nullable_schema(prop)
    return transformed


def _gemini_function_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        transformed: dict[str, Any] = {}
        for key, value in schema.items():
            if key == 'additionalProperties':
                continue
            transformed[key] = _gemini_function_schema(value)
        return transformed
    if isinstance(schema, list):
        return [_gemini_function_schema(item) for item in schema]
    return schema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    runner: "ToolRunner"

    def generic_function_declaration(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters_schema,
        }

    def openai_tool(self) -> dict[str, Any]:
        declaration = self.generic_function_declaration()
        return {
            'type': 'function',
            'name': declaration['name'],
            'description': declaration['description'],
            'parameters': _openai_strict_schema(declaration['parameters']),
            'strict': True,
        }

    def gemini_function_declaration(self) -> dict[str, Any]:
        declaration = self.generic_function_declaration()
        return {
            'name': declaration['name'],
            'description': declaration['description'],
            'parameters': _gemini_function_schema(declaration['parameters']),
        }


class ToolRunner(Protocol):
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...
