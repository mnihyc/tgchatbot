from __future__ import annotations

"""Prompt assembly.

The runtime keeps the user-provided preset/system prompt as the anchor.
This module only adds optional operational guidance around that anchor.
For users who want a raw preset with no extra framework text, use
PromptInjectionMode.EXACT.
"""

from tgchatbot.domain.models import ChatMode, PromptInjectionMode, SessionSettings, StickerMode


def build_system_prompt(settings: SessionSettings) -> str:
    custom = settings.system_prompt.strip()
    if settings.prompt_injection_mode == PromptInjectionMode.EXACT:
        return custom or SessionSettings().system_prompt

    mode_guidance = {
        ChatMode.CHAT: ('Preserving the personality. Do not call tools.'),
        ChatMode.ASSIST: ('Preserving the personality. Use tools sparingly when they materially improve correctness or utility.'),
        ChatMode.AGENT: ('Preserving the personality. Prefer tool use over guessing, and stop once the request is satisfied.'),
    }[settings.mode]

    memory_guidance = ('This conversation is persistent across interactions. Keep recent working context verbatim when practical.'
                       'Older context may be represented by durable episode memory blocks and auxiliary digest memory blocks; do not assume every memory block is equally authoritative.')
    metadata_guidance = ('Some user messages may begin with an automatic metadata (sender) header'
                         'in the form [username=<handle> nickname="<display name>" time=<local timestamp>] or similar.'
                         'Treat these headers as transport context, not as text the user wrote. ')
    style_guidance = 'Respond in either plaintext or Markdown. '
    attachment_guidance = ('Images, stickers, and files in message history are part of the ongoing conversation context.'
                           'Earlier files may change after shell or Python edits.') if settings.mode != ChatMode.CHAT else ''
    tool_guidance = ('Use browsing and tool execution only when they materially improve correctness or utility.'
                     'Avoid unnecessary browsing, repeated searches, or needless artifact generation.'
                     'Additional tools may include:'
                     '   - shell_exec: run POSIX sh commands in the persistent remote session workspace'
                     '   - python_exec: run Python 3 code in the persistent remote session workspace'
                     '   - file_send: send a remote workspace file back to the user in chat'
                     'Remote execution environment:'
                     '   - Current working directory is the session workspace root unless cwd_subdir is provided'
                     '   - Shell commands run with sh, not bash'
                     '   - Python runs with python3'
                     '   - Environment variables:'
                     '     TGCHATBOT_SESSION_DIR, TGCHATBOT_INPUT_DIR, TGCHATBOT_OUTPUT_DIR'
                     '   - User-uploaded files is mirrored into TGCHATBOT_INPUT_DIR'
                     '   - Prefer shell_exec for file inspection and CLI tasks; prefer python_exec for structured parsing, analysis, and transformations'
                     ) if settings.mode != ChatMode.CHAT else ''
    sticker_guidance = ''
    if settings.sticker_mode == StickerMode.AUTO:
        sticker_guidance = (
            'Two sticker tools are available: sticker_query and sticker_send_selected. Use them sparingly when a sticker adds a precise social reaction that plain text alone would not. '
            'Query first, inspect the ranked shortlist, then explicitly send one sticker by sticker_id. '
            'When querying, always provide intent_core and usually stop there. Add at most one or two simple helper hints only when they clearly matter: emotion_tone, social_goal, visual_hint, text_hint, prefer_pack, or prefer_cluster. '
            'Think in this order when choosing stickers: what social message it sends, what hidden subtext or implication it carries, what face and pose deliver that, what visual family it should stay in, and only then how different it should be from recent stickers. '
            'Use persona when the sticker should keep a recurring visual family or expressive bias across the session. '
            'Use selection_lens when subtle human factors matter, such as social read, hidden implication, face-and-pose delivery, continuity note, or avoid_misread_as. '
            'Use diversity_preference=prefer_fresh_variant only when the reaction is close to a recent one but a slightly different suitable variant is preferable. '
            'Use advanced only when you intentionally need axis-level control such as semantic_focus, visual_focus, text_constraints, or style_focus.style_goal. '
            'Do not put usernames, bot names, mentions, ids, paths, or tool names into sticker intent fields. '
            'Inspect selection_summary, social_read, expression_fit, persona_fit, continuity_note, fit_signals, and warnings first. Use candidate.debug.score_breakdown only when you need to compare close alternatives. '
            'Default to static stickers unless allow_animation=true or advanced.intensity_limits.allow_animation=true is materially better.'
        )


    return '\n\n'.join(part for part in [custom, mode_guidance, memory_guidance, metadata_guidance, style_guidance, attachment_guidance, tool_guidance, sticker_guidance] if part.strip())


def build_compaction_prompt(*, mode: str) -> str:
    if mode not in {'toolspan', 'episode', 'digest'}:
        raise ValueError(f'unsupported compaction mode: {mode}')
    role_line = (
        'normalized tool-heavy raw chat span with complete tool cycles and any interleaved user or assistant turns'
        if mode == 'toolspan'
        else 'normalized raw chat history plus any earlier L0 tool-span blocks that sit inside the slice'
        if mode == 'episode'
        else 'adjacent sealed episode memory blocks with metadata'
    )
    goal_line = (
        'Create a faithful L0 tool-span block that still reads correctly when the source mixes problem-solving with ordinary sharing or reaction turns.'
        if mode == 'toolspan'
        else 'Create a faithful L1 episode block that works for both task-solving exchanges and ordinary chatting or sharing without flattening them into the same style.'
        if mode == 'episode'
        else 'Create a faithful L2 digest block that reconciles earlier episodes from either task-solving or ordinary chat/sharing threads into durable memory.'
    )
    sections = {
        'toolspan': 'scope, interaction_mode, participants, topics, user_profile, user_intent_or_shared_context, why_it_mattered, assistant_strategy, tool_timeline, results_or_takeaways, decisions, open_loops, artifacts, uncertainties, retained_raw_excerpts',
        'episode': 'scope, interaction_mode, tool_usage, participants, topics, user_profile, user_intent_or_shared_context, why_it_mattered, interaction_timeline, results_or_takeaways, decisions, open_loops, artifacts, uncertainties, retained_raw_excerpts',
        'digest': 'scope, interaction_modes_seen, participants, topics, user_profile, recurring_requests_or_shared_threads, why_history_matters_now, durable_state, important_changes, decisions, open_loops, artifacts, uncertainties, parent_refs',
    }[mode]
    common_lines = [
        'Use the actual participant username or a stable short label instead of repeating generic user when different people played different roles. If involved, keep effective individual users separated and use more bullet points to describe each. ',
        'There are two common source situations: (a) task_execution: the user is trying to solve, inspect, fix, or produce something, often with tool use; (b) chat_or_sharing: the user is chatting normally, reacting, telling a story, or sharing something, often without a concrete task.',
        'Choose interaction_mode carefully. Use mixed only when both situations materially coexist in the same slice.',
        'For user_intent_or_shared_context: in task_execution, record the ask or problem to solve; in chat_or_sharing, record what and which user shared, reacted to, wanted acknowledged, or wanted the assistant to keep in mind.',
        'For why_it_mattered: in task_execution, record motivation, stakes, blockers, deadlines, or success criteria; in chat_or_sharing, record emotional significance, relationship context, reason for sharing, or why the topic mattered. Leave empty rather than inventing detail.',
        'For results_or_takeaways: in task_execution, record the answer, fix, output, or verified conclusion; in chat_or_sharing, record the main takeaway, updated understanding, social outcome, or memory-worthy point. It is fine if the result is simply that the user shared an update and no action was required.',
    ]
    field_guidance = {
        'toolspan': [
            'scope: one sentence naming the span plus why it mattered.',
            'assistant_strategy: record the assistant approach, retries, pivots, refusal boundaries, or moderation choices. Keep this about strategy, not raw output.',
            'tool_timeline: chronological bullets using `tool or actor -> purpose -> meaningful result or obstacle`.',
            'results_or_takeaways: only end-state facts or takeaways by the end of the span.',
            'Use tool_timeline for tool actions and results_or_takeaways for what those actions achieved. Do not dump stdout or code contents.',
        ],
        'episode': [
            'scope: one sentence naming the slice plus why it mattered.',
            'interaction_timeline: chronological bullets using `actor -> action -> consequence` and make it work for both task-solving and ordinary conversation.',
            'results_or_takeaways: only what became true, was learned, or was worth remembering by the end of the slice.',
            'For normal group-style chatting, keep the same section structure: what the user brought up, why it mattered, how others responded, and what should be remembered later.',
        ],
        'digest': [
            'scope: one sentence naming the reconciled shard plus why it still matters now.',
            'interaction_modes_seen: include one or more of task_execution, chat_or_sharing, mixed without duplicates.',
            'recurring_requests_or_shared_threads: capture repeated asks, recurring projects, or recurring conversational topics that span parents.',
            'why_history_matters_now: explain the active dependency or rationale that makes this shard still relevant.',
            'durable_state: only state that should survive beyond one episode.',
            'important_changes: version changes, reversals, mood or stance changes, or meaningful progress across episodes.',
        ],
    }[mode]
    anti_patterns = [
        'Do not collapse multiple people into one generic user. ',
        'Do not merely restate tool output, code contents, or quoted chat lines. Center the interaction: user side, why it mattered, what happened, outcome, and future relevance.',
        'Do not copy transport wrappers like [Auto-generated bot message, do not reply.] into output.',
        'Do not copy raw provider or tool protocol, raw JSON, call IDs, or stdout/stderr dumps unless the exact text must be retained later.',
        'Time bounds are supplied separately through compaction metadata. Do not invent vague time prose such as Unknown.',
        'Use user_profile only for durable user preferences, recurring constraints, stable environment facts, or habits likely to matter later.',
        'If the source only inspected something, keep that in the timeline and takeaways. Do not promote bounded inspection into timeless fact unless the source confirms it remains true afterward.',
        'retained_raw_excerpts is optional and capped at three short literal excerpts. Leave it empty unless exact wording materially matters later.',
    ]
    fewshot = {
        'toolspan': '''Few-shot patterns:
Task-style example:
{
  "scope": "Tool-heavy span where the user wanted a replay-payload bug fixed so the bot could stop failing at runtime.",
  "interaction_mode": "task_execution",
  "participants": ["user: name", "user: name2", "assistant", "tool: name"],
  "topics": ["openai replay", "reasoning items", "payload repair"],
  "user_profile": ["User prefers narrow fixes over broad redesigns."],
  "user_intent_or_shared_context": ["Repair the replay payload bug in the latest patch."],
  "why_it_mattered": ["The bot was failing with 400 errors and needed a minimal corrective patch."],
  "assistant_strategy": ["Traced persisted reasoning items, patched replay serialization, and kept encrypted reasoning handles while dropping clear-text summaries."],
  "tool_timeline": [
    "file inspection -> locate OpenAI replay payload builder -> confirmed reasoning items were missing required summary field",
    "code patch -> keep encrypted_content and send summary as an empty list -> preserves API shape without extra token cost"
  ],
  "results_or_takeaways": ["Replay payloads now include reasoning summary as an empty list and stop triggering the missing-parameter error."],
  "decisions": ["Keep encrypted reasoning handles but omit clear-text reasoning summaries from reply-time history."],
  "open_loops": [],
  "artifacts": ["tgchatbot/providers/openai_responses.py", "tgchatbot/core/runtime.py", "reasoning.summary=[]"],
  "uncertainties": [],
  "retained_raw_excerpts": []
}
Mixed example:
{
  "scope": "Tool-heavy span where the user shared a log excerpt and also wanted the assistant to inspect it and explain the failure path.",
  "interaction_mode": "mixed",
  "participants": ["user: name", "user: name2", "assistant", "tool: name"],
  "topics": ["log analysis", "runtime failure", "explanation"],
  "user_profile": [],
  "user_intent_or_shared_context": ["The user(name) shared runtime logs.", "The user(name) wanted the assistant to reconstruct the failure path from those logs."],
  "why_it_mattered": ["The shared logs were the only concrete evidence for the regression and the user wanted a precise explanation rather than guesswork."],
  "assistant_strategy": ["Read the logs first, reconstructed the sequence, then explained the concrete failure mechanism."],
  "tool_timeline": ["file inspection -> read uploaded logs -> extracted the failing call chain"],
  "results_or_takeaways": ["The assistant identified the concrete runtime failure path from the shared logs."],
  "decisions": ["Base the explanation on the uploaded evidence rather than inference alone."],
  "open_loops": [],
  "artifacts": ["uploaded log file", "failing call chain"],
  "uncertainties": [],
  "retained_raw_excerpts": []
}''',
        'episode': '''Few-shot patterns:
Task-style example:
{
  "scope": "Interaction slice where the user asked for a diff between two patches so they could inspect exactly what changed and why.",
  "interaction_mode": "task_execution",
  "tool_usage": [
    "user(name) -> requested file inspection -> tool(shell_exec) locate OpenAI replay payload builder -> confirmed reasoning items were missing required summary field",
    "user(name) -> requested code patch -> keep encrypted_content and send summary as an empty list -> preserves API shape without extra token cost -> tool(shell_exec) finished patch"
  ],
  "participants": ["user: name", "user: name2", "assistant"],
  "topics": ["patch diff", "regression triage", "verification"],
  "user_profile": ["User(name) wants precise, inspectable patch artifacts rather than verbal assurances."],
  "user_intent_or_shared_context": ["Produce the diff between the previous patch and the new patch."],
  "why_it_mattered": ["The user wanted to verify the exact changes before trusting the fix."],
  "interaction_timeline": [
    "user(name) -> requested a diff for the newest patch pair -> narrowed the task to artifact comparison",
    "assistant -> generated a unified diff and surfaced only the meaningful source-file changes -> gave the user something directly auditable"
  ],
  "results_or_takeaways": ["A unified diff artifact was produced for the requested patch pair."],
  "decisions": ["Report only the meaningful source changes and treat compiled bytecode differences as rebuild artifacts."],
  "open_loops": [],
  "artifacts": ["unified diff artifact", "changed source file list"],
  "uncertainties": [],
  "retained_raw_excerpts": []
}
Chat/sharing example:
{
  "scope": "Conversation slice where the user shared an update and the assistant mainly acknowledged, clarified, and preserved the important context.",
  "interaction_mode": "chat_or_sharing",
  "tool_usage": [],
  "participants": ["user: name", "user: name2", "assistant"],
  "topics": ["status update", "conversation continuity", "shared context"],
  "user_profile": [],
  "user_intent_or_shared_context": ["The user(name) shared a status update rather than asking for a concrete task."],
  "why_it_mattered": ["The update was worth preserving because it changed the conversation context going forward."],
  "interaction_timeline": [
    "user(name) -> shared an update or observation -> introduced new conversational context",
    "assistant -> acknowledged and clarified the important point -> preserved what should matter later"
  ],
  "results_or_takeaways": ["The important shared context was captured without turning it into a fake task or a tool-output summary."],
  "decisions": [],
  "open_loops": [],
  "artifacts": [],
  "uncertainties": [],
  "retained_raw_excerpts": []
}''',
        'digest': '''Few-shot pattern:
{
  "scope": "Digest of earlier runtime-compaction episodes that still matters because the same conversation-memory pipeline remains under active revision.",
  "interaction_modes_seen": ["task_execution", "chat_or_sharing"],
  "participants": ["user: name", "user: name2", "assistant"],
  "topics": ["compaction", "runtime regressions", "patch iteration"],
  "user_profile": ["User(name) expects design changes to be reflected in concrete code patches and dislikes hidden heuristic logic."],
  "recurring_requests_or_shared_threads": ["Keep the compaction design aligned with the stated architecture while removing misleading or low-quality logic."],
  "why_history_matters_now": ["The same compaction pipeline is still being revised, so earlier regressions, decisions, and conversational expectations remain relevant to new patches."],
  "durable_state": ["Compaction uses structured-output schemas for L0, L1, and L2 blocks and stores deterministic block metadata separately from rendered text."],
  "important_changes": ["The design moved away from heuristic text mining and toward explicit structured fields plus deterministic timestamps."],
  "decisions": ["Prefer schema and prompt improvements over validator-enforced quality rules."],
  "open_loops": ["Further prompt and schema tuning may still be needed if blocks continue to miss user-side context."],
  "artifacts": ["compaction schemas", "runtime selection logic", "provider structured-output integration"],
  "uncertainties": [],
  "parent_refs": ["L1#10", "L1#11", "L1#12"]
}''',
    }[mode]
    return '\n'.join([
        'You are writing durable memory for a long-running chat system.',
        f'Input consists of {role_line}.',
        goal_line,
        f'Populate these sections faithfully: {sections}.',
        'Fill each field deliberately:',
        *[f'- {item}' for item in common_lines],
        *[f'- {item}' for item in field_guidance],
        *[f'- {item}' for item in anti_patterns],
        'Return exactly one JSON object that matches the provided structured-output schema and no surrounding commentary.',
        'Use concise factual bullets. Prefer short lists over long narrative paragraphs.',
        'Good output preserves both the user side and the assistant side: what the user wanted or shared, why it mattered, what happened, what changed, and what still matters now.',
        fewshot,
    ]).strip()
