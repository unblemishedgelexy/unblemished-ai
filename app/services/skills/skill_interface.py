from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.core.logger import StructuredLogger
from app.repositories.skill_repository import SkillRepository, SkillRow
from app.services.tools.tool_interface import ToolInterface
from app.utils.helpers import utc_now


@dataclass(slots=True)
class SkillMatchResult:
    skill_id: str
    trigger_text: str
    tool_name: str
    tool_invocation: dict[str, Any]
    draft_answer: str


@dataclass(slots=True)
class SkillLearningResult:
    learned: bool
    active: bool
    skill_id: str
    trigger_text: str
    tool_name: str
    correction_count: int
    reason: str = ""


@dataclass(slots=True)
class SkillDefinition:
    skill_id: str
    user_id: str
    trigger_text: str
    trigger_type: str
    tool_name: str
    tool_arguments: dict[str, Any]
    correction_count: int
    source: str
    active: bool
    created_at: datetime
    updated_at: datetime


class SkillInterface:
    def __init__(
        self,
        *,
        repository: SkillRepository,
        tool_interface: ToolInterface,
        logger: StructuredLogger,
        learning_threshold: int = 2,
        max_match_scan: int = 100,
    ) -> None:
        self._repository = repository
        self._tool_interface = tool_interface
        self._logger = logger
        self._learning_threshold = max(learning_threshold, 2)
        self._max_match_scan = max(max_match_scan, 10)

    async def match_and_execute(
        self,
        *,
        user_id: str,
        trace_id: str,
        user_input: str,
        context: dict[str, Any],
    ) -> SkillMatchResult | None:
        await self._repository.initialize()

        skills = await self._repository.list_active_skills(
            user_id=user_id,
            limit=self._max_match_scan,
        )
        if not skills:
            return None

        for skill in skills:
            if not _trigger_matches(
                trigger_text=skill.trigger_text,
                trigger_type=skill.trigger_type,
                user_input=user_input,
            ):
                continue
            if not self._tool_interface.is_tool_allowed(skill.tool_name):
                self._logger.warning(
                    "skill.match.rejected",
                    trace_id=trace_id,
                    user_id=user_id,
                    memory_id="n/a",
                    retrieval_count=0,
                    skill_id=skill.skill_id,
                    reason="tool_not_allowed",
                    tool_name=skill.tool_name,
                )
                continue

            arguments = _render_tool_arguments(
                template=skill.tool_arguments,
                user_input=user_input,
                context=context,
            )
            if not _arguments_safe(arguments):
                self._logger.warning(
                    "skill.match.rejected",
                    trace_id=trace_id,
                    user_id=user_id,
                    memory_id="n/a",
                    retrieval_count=0,
                    skill_id=skill.skill_id,
                    reason="unsafe_tool_arguments",
                    tool_name=skill.tool_name,
                )
                continue

            invocation = await self._tool_interface.invoke(
                tool_name=skill.tool_name,
                arguments=arguments,
                trace_id=trace_id,
                user_id=user_id,
            )
            self._logger.info(
                "skill.match.executed",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                skill_id=skill.skill_id,
                tool_name=skill.tool_name,
                success=bool(invocation.get("success", False)),
            )
            return SkillMatchResult(
                skill_id=skill.skill_id,
                trigger_text=skill.trigger_text,
                tool_name=skill.tool_name,
                tool_invocation=invocation,
                draft_answer=_build_skill_draft_answer(
                    trigger_text=skill.trigger_text,
                    tool_name=skill.tool_name,
                    invocation=invocation,
                ),
            )
        return None

    async def learn_from_correction(
        self,
        *,
        user_id: str,
        trace_id: str,
        user_input: str,
    ) -> SkillLearningResult | None:
        await self._repository.initialize()
        parsed = _parse_learning_instruction(user_input)
        if parsed is None:
            return None

        trigger_text = parsed["trigger_text"]
        tool_name = parsed["tool_name"]
        if not self._tool_interface.is_tool_allowed(tool_name):
            self._logger.warning(
                "skill.learn.rejected",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                trigger_text=trigger_text,
                tool_name=tool_name,
                reason="tool_not_allowed",
            )
            return SkillLearningResult(
                learned=False,
                active=False,
                skill_id="n/a",
                trigger_text=trigger_text,
                tool_name=tool_name,
                correction_count=0,
                reason="tool_not_allowed",
            )

        template = _default_arguments_for_tool(tool_name)
        if not _arguments_safe(template):
            self._logger.warning(
                "skill.learn.rejected",
                trace_id=trace_id,
                user_id=user_id,
                memory_id="n/a",
                retrieval_count=0,
                trigger_text=trigger_text,
                tool_name=tool_name,
                reason="unsafe_tool_arguments",
            )
            return SkillLearningResult(
                learned=False,
                active=False,
                skill_id="n/a",
                trigger_text=trigger_text,
                tool_name=tool_name,
                correction_count=0,
                reason="unsafe_tool_arguments",
            )

        existing = await self._repository.get_skill(
            user_id=user_id,
            trigger_text=trigger_text,
            tool_name=tool_name,
        )
        now = utc_now()
        if existing is None:
            correction_count = 1
            created_at = now
            skill_id = str(uuid4())
        else:
            correction_count = existing.correction_count + 1
            created_at = existing.created_at
            skill_id = existing.skill_id
        active = correction_count >= self._learning_threshold

        row = SkillRow(
            skill_id=skill_id,
            user_id=user_id,
            trigger_text=trigger_text,
            trigger_type="contains",
            tool_name=tool_name,
            tool_arguments=template,
            correction_count=correction_count,
            source="user_correction",
            active=active,
            created_at=created_at,
            updated_at=now,
        )
        await self._repository.upsert_skill(row)
        self._logger.info(
            "skill.learn.recorded",
            trace_id=trace_id,
            user_id=user_id,
            memory_id="n/a",
            retrieval_count=0,
            skill_id=skill_id,
            trigger_text=trigger_text,
            tool_name=tool_name,
            correction_count=correction_count,
            active=active,
        )
        return SkillLearningResult(
            learned=True,
            active=active,
            skill_id=skill_id,
            trigger_text=trigger_text,
            tool_name=tool_name,
            correction_count=correction_count,
            reason="stored",
        )

    async def list_skills(
        self,
        *,
        user_id: str,
        include_inactive: bool = True,
        limit: int = 200,
    ) -> list[SkillDefinition]:
        await self._repository.initialize()
        rows = await self._repository.list_skills(
            user_id=user_id,
            include_inactive=include_inactive,
            limit=max(limit, 1),
        )
        return [_to_definition(row) for row in rows]

    async def upsert_skill(
        self,
        *,
        user_id: str,
        trigger_text: str,
        trigger_type: str,
        tool_name: str,
        tool_arguments: dict[str, Any] | None,
        active: bool,
        source: str = "dashboard",
    ) -> SkillDefinition:
        await self._repository.initialize()
        normalized_trigger = _normalize_trigger(trigger_text)
        if not normalized_trigger:
            raise ValueError("invalid_trigger_text")

        normalized_tool = _normalize_tool_name(tool_name)
        if not normalized_tool:
            raise ValueError("invalid_tool_name")
        if not self._tool_interface.is_tool_allowed(normalized_tool):
            raise PermissionError("tool_not_allowed")

        normalized_type = trigger_type.strip().lower()
        if normalized_type not in {"contains", "exact"}:
            raise ValueError("invalid_trigger_type")

        arguments = tool_arguments or _default_arguments_for_tool(normalized_tool)
        if not _arguments_safe(arguments):
            raise ValueError("unsafe_tool_arguments")

        existing = await self._repository.get_skill(
            user_id=user_id,
            trigger_text=normalized_trigger,
            tool_name=normalized_tool,
        )
        now = utc_now()
        row = SkillRow(
            skill_id=existing.skill_id if existing is not None else str(uuid4()),
            user_id=user_id,
            trigger_text=normalized_trigger,
            trigger_type=normalized_type,
            tool_name=normalized_tool,
            tool_arguments=arguments,
            correction_count=max(existing.correction_count, 1) if existing is not None else 1,
            source=source,
            active=active,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        await self._repository.upsert_skill(row)
        return _to_definition(row)

    async def delete_skill(
        self,
        *,
        skill_id: str,
        user_id: str | None = None,
    ) -> bool:
        await self._repository.initialize()
        existing = await self._repository.get_skill_by_id(skill_id=skill_id)
        if existing is None:
            return False
        if user_id is not None and existing.user_id != user_id:
            return False
        return await self._repository.delete_skill(skill_id=skill_id)

    async def is_ready(self) -> bool:
        return await self._repository.is_ready()

    async def count_skills(
        self,
        *,
        user_id: str,
        include_inactive: bool = True,
        limit: int = 5000,
    ) -> int:
        skills = await self.list_skills(
            user_id=user_id,
            include_inactive=include_inactive,
            limit=limit,
        )
        return len(skills)


def _build_skill_draft_answer(
    *,
    trigger_text: str,
    tool_name: str,
    invocation: dict[str, Any],
) -> str:
    success = bool(invocation.get("success", False))
    status = "success" if success else "failed"
    result_summary = str(invocation.get("result_summary", "No tool result summary."))
    return (
        "Structured Reasoning Output\n"
        f"- Direct Answer: {result_summary}\n"
        f"- Skill Trigger Match: '{trigger_text}'\n"
        f"- Skill Action: Executed tool '{tool_name}' ({status})."
    )


def _trigger_matches(*, trigger_text: str, trigger_type: str, user_input: str) -> bool:
    normalized_trigger = trigger_text.strip().lower()
    normalized_input = user_input.strip().lower()
    if not normalized_trigger or not normalized_input:
        return False
    if trigger_type == "exact":
        return normalized_input == normalized_trigger
    return normalized_trigger in normalized_input


def _parse_learning_instruction(user_input: str) -> dict[str, str] | None:
    for pattern in _LEARNING_PATTERNS:
        match = pattern.search(user_input)
        if match is None:
            continue
        trigger_text = _normalize_trigger(match.group("trigger"))
        tool_name = _normalize_tool_name(match.group("tool"))
        if not trigger_text or not tool_name:
            continue
        return {"trigger_text": trigger_text, "tool_name": tool_name}
    return None


def _normalize_trigger(raw: str) -> str:
    value = raw.strip().strip("\"'")
    value = re.sub(r"\s+", " ", value)
    if len(value) < 3:
        return ""
    return value.lower()


def _normalize_tool_name(raw: str) -> str:
    value = raw.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{2,64}", value):
        return ""
    return value


def _default_arguments_for_tool(tool_name: str) -> dict[str, Any]:
    if tool_name == "keyword_extract":
        return {"text": "{{user_input}}", "max_keywords": 6}
    if tool_name == "context_digest":
        return {"context": "{{context}}"}
    if tool_name == "priority_estimator":
        return {"goal": "{{user_input}}", "priority_hint": "{{context.priority}}"}
    return {}


def _render_tool_arguments(
    *,
    template: dict[str, Any],
    user_input: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in template.items():
        rendered[key] = _render_template_value(
            value=value,
            user_input=user_input,
            context=context,
        )
    return rendered


def _render_template_value(*, value: Any, user_input: str, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _render_template_value(value=item, user_input=user_input, context=context)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_template_value(value=item, user_input=user_input, context=context) for item in value]
    if not isinstance(value, str):
        return value
    if value == "{{user_input}}":
        return user_input
    if value == "{{context}}":
        return context
    if value == "{{context.priority}}":
        hint = context.get("priority")
        return hint if isinstance(hint, str) and hint.strip() else None
    return value.replace("{{user_input}}", user_input)


def _arguments_safe(arguments: dict[str, Any]) -> bool:
    blocked_key_tokens = {"cmd", "command", "shell", "script", "powershell", "bash", "python", "exec"}
    blocked_value_tokens = {"rm -rf", "powershell", "bash -c", "cmd /c", "os.system", "subprocess"}

    def _visit(node: Any) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key).lower()
                if any(token in key_text for token in blocked_key_tokens):
                    return False
                if not _visit(value):
                    return False
            return True
        if isinstance(node, list):
            return all(_visit(item) for item in node)
        if isinstance(node, str):
            lowered = node.lower()
            if len(lowered) > 4000:
                return False
            if any(token in lowered for token in blocked_value_tokens):
                return False
            return True
        if isinstance(node, (int, float, bool)) or node is None:
            return True
        return False

    return _visit(arguments)


_LEARNING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"""when\s+i\s+say\s+["']?(?P<trigger>[^"']+?)["']?\s*,?\s*(?:please\s*)?(?:use|run|call)\s+tool\s*:?\s*(?P<tool>[a-zA-Z0-9_]+)""",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"""trigger\s*[:=]\s*["']?(?P<trigger>[^"'\n,;]+)["']?\s*(?:->|,)?\s*(?:use|run|call)?\s*tool\s*[:=]\s*(?P<tool>[a-zA-Z0-9_]+)""",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"""learn\s+skill\s*[:\-]\s*["']?(?P<trigger>[^"']+?)["']?\s*=>\s*(?P<tool>[a-zA-Z0-9_]+)""",
        flags=re.IGNORECASE,
    ),
)


def _to_definition(row: SkillRow) -> SkillDefinition:
    return SkillDefinition(
        skill_id=row.skill_id,
        user_id=row.user_id,
        trigger_text=row.trigger_text,
        trigger_type=row.trigger_type,
        tool_name=row.tool_name,
        tool_arguments=dict(row.tool_arguments),
        correction_count=row.correction_count,
        source=row.source,
        active=row.active,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
