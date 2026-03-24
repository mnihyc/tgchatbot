from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _CompactionModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


InteractionMode = Literal['task_execution', 'chat_or_sharing', 'mixed']


class ToolSpanMemoryBlock(_CompactionModel):
    scope: str = Field(description='One concise sentence naming this tool-heavy span and why it mattered.')
    interaction_mode: InteractionMode = Field(description='Choose task_execution for solving or fixing work, chat_or_sharing for casual sharing or reaction-first exchange, or mixed when both happened.')
    participants: list[str] = Field(description='Only people or systems that materially affected the span or its outcome.')
    topics: list[str] = Field(description='2 to 6 short retrieval-oriented labels, not full sentences.')
    user_profile: list[str] = Field(description='Only durable user preferences, stable environment facts, recurring constraints, or habits likely to matter later.')
    user_intent_or_shared_context: list[str] = Field(description='For task_execution, list the ask or problem to solve. For chat_or_sharing, list what the user shared, reacted to, or wanted acknowledged. For mixed, include both clearly.')
    why_it_mattered: list[str] = Field(description='Motivation, success criteria, stakes, deadlines, blockers, emotional significance, or relationship context. Leave empty if not supported; do not repeat user_intent_or_shared_context.')
    assistant_strategy: list[str] = Field(description='How the assistant approached the span: plan, retry, pivot, refusal, moderation choice, or explicit decision boundary.')
    tool_timeline: list[str] = Field(description='Ordered bullets in the form `tool or actor -> purpose -> meaningful result or obstacle`. Keep it chronological and selective.')
    results_or_takeaways: list[str] = Field(description='Concrete end state, answer, fix, conclusion, or memory-worthy takeaway by the end of the span.')
    decisions: list[str] = Field(description='Chosen path, commitment, or next-step decision that should influence later turns.')
    open_loops: list[str] = Field(description='What still needs follow-up, confirmation, or later work.')
    artifacts: list[str] = Field(description='Retrieval-worthy files, repos, URLs, commands, environments, tools, versions, or exact error identifiers.')
    uncertainties: list[str] = Field(description='Ambiguities, conflicts, or evidence gaps. Use when something could not be confirmed.')
    retained_raw_excerpts: list[str] = Field(default_factory=list, max_length=3, description='At most three short literal excerpts. Use only when exact wording materially matters later.')


class EpisodeMemoryBlock(_CompactionModel):
    scope: str = Field(description='One concise sentence naming this interaction slice and why it mattered.')
    interaction_mode: InteractionMode = Field(description='Choose task_execution for solving or fixing work, chat_or_sharing for normal chatting or sharing, or mixed when both happened in the same slice.')
    tool_usage: list[str] = Field(default_factory=list, description='Short bullets summarizing meaningful tool usage from raw messages or carried forward from parent L0 blocks, if any. Focus on user -> purpose -> tool -> important result, not raw output.')
    participants: list[str] = Field(description='Only people or systems that materially shaped the interaction or outcome.')
    topics: list[str] = Field(description='2 to 6 short retrieval-oriented labels, not full sentences.')
    user_profile: list[str] = Field(description='Only durable user preferences, stable environment facts, recurring constraints, or habits likely to matter later.')
    user_intent_or_shared_context: list[str] = Field(description='For task_execution, list the ask or problem. For chat_or_sharing, list what the user shared, reacted to, or wanted acknowledged. For mixed, keep the task and sharing context separate.')
    why_it_mattered: list[str] = Field(description='Motivation, desired outcome, stakes, blocker, emotional significance, or social reason for sharing. Leave empty if not supported; do not repeat user_intent_or_shared_context.')
    interaction_timeline: list[str] = Field(description='Ordered bullets in the form `actor -> action -> consequence`. Preserve who did what and why it mattered. This must work for both task-solving and casual or share-first interactions.')
    results_or_takeaways: list[str] = Field(description='Concrete end state, answer, resolution, social takeaway, or memory-worthy point by the end of the slice.')
    decisions: list[str] = Field(description='Chosen path, commitment, or next-step decision that should influence later turns.')
    open_loops: list[str] = Field(description='What still needs follow-up, confirmation, or later work.')
    artifacts: list[str] = Field(description='Retrieval-worthy files, repos, URLs, tools, commands, environments, or exact error identifiers.')
    uncertainties: list[str] = Field(description='Ambiguities, conflicts, or evidence gaps. Use when something could not be confirmed.')
    retained_raw_excerpts: list[str] = Field(default_factory=list, max_length=3, description='At most three short literal excerpts. Use only when exact wording materially matters later.')


class DigestMemoryBlock(_CompactionModel):
    scope: str = Field(description='One concise sentence naming what earlier shard this digest reconciles and why it still matters now.')
    interaction_modes_seen: list[InteractionMode] = Field(description='Interaction modes that appear across the parent episodes, in rough importance order without duplicates.')
    participants: list[str] = Field(description='Only people or systems that remain materially relevant across the shard.')
    topics: list[str] = Field(description='2 to 6 short retrieval-oriented labels for this digest shard.')
    user_profile: list[str] = Field(description='Only durable user preferences, stable environment facts, recurring constraints, or habits that still matter across the shard.')
    recurring_requests_or_shared_threads: list[str] = Field(description='Repeated user goals, asks, shared topics, or conversational threads that recur across the shard.')
    why_history_matters_now: list[str] = Field(description='Why these episodes belong together now: active dependency, continuing rationale, or still-relevant personal or project context.')
    durable_state: list[str] = Field(description='Durable state or facts that should survive beyond individual episodes.')
    important_changes: list[str] = Field(description='Meaningful updates, reversals, tone changes, or progress across the shard.')
    decisions: list[str] = Field(description='Important commitments, choices, or next-step decisions still relevant now.')
    open_loops: list[str] = Field(description='Important unresolved items that still matter after reconciliation.')
    artifacts: list[str] = Field(description='Retrieval-worthy files, repos, URLs, tools, commands, environments, or exact error identifiers.')
    uncertainties: list[str] = Field(description='Ambiguities, conflicts, or evidence gaps that remain after reconciliation.')
    parent_refs: list[str] = Field(description='Parent episode block refs such as L1#12. Keep chronological order.')


_COMPACTION_MODELS: dict[str, type[_CompactionModel]] = {
    'toolspan': ToolSpanMemoryBlock,
    'episode': EpisodeMemoryBlock,
    'digest': DigestMemoryBlock,
}

_SCHEMA_NAMES: dict[str, str] = {
    'toolspan': 'toolspan_memory_block',
    'episode': 'episode_memory_block',
    'digest': 'digest_memory_block',
}


def compaction_model_for_mode(mode: str) -> type[_CompactionModel]:
    try:
        return _COMPACTION_MODELS[mode]
    except KeyError as exc:
        raise ValueError(f'unsupported compaction mode: {mode}') from exc


def compaction_schema_name(mode: str) -> str:
    try:
        return _SCHEMA_NAMES[mode]
    except KeyError as exc:
        raise ValueError(f'unsupported compaction mode: {mode}') from exc


def compaction_json_schema(mode: str) -> dict[str, Any]:
    schema = compaction_model_for_mode(mode).model_json_schema()
    if isinstance(schema, dict):
        schema.setdefault('type', 'object')
        schema.setdefault('additionalProperties', False)
        properties = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        schema['required'] = list(properties.keys())
    return schema


def parse_structured_candidate(mode: str, payload: Any) -> dict[str, Any]:
    model = compaction_model_for_mode(mode)
    parsed = model.model_validate(payload)
    return parsed.model_dump(mode='python')


__all__ = [
    'ToolSpanMemoryBlock',
    'EpisodeMemoryBlock',
    'DigestMemoryBlock',
    'compaction_model_for_mode',
    'compaction_schema_name',
    'compaction_json_schema',
    'parse_structured_candidate',
    'ValidationError',
]
