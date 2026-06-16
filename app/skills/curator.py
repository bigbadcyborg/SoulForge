"""Curator: review active skills for staleness, duplicates, and bloat."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime
from app.skills.skill_crystallizer import validate_skill_content
from app.skills.skill_manager import SkillManager

CURATOR_COMPLETION_OVERRIDES = {
    "temperature": 0.2,
    "max_tokens": 2048,
    "stop": ["</s>"],
}

FINDING_STALE = "stale"
FINDING_BLOAT = "bloat"
FINDING_DUPLICATE = "duplicate"

ACTION_ARCHIVE = "archive"
ACTION_COMPACT = "compact"


@dataclass
class CuratorFinding:
    finding_id: str
    finding_type: str
    skill_name: str
    rationale: str
    proposed_action: str
    proposed_content: str = ""
    related_skill: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "skill_name": self.skill_name,
            "rationale": self.rationale,
            "proposed_action": self.proposed_action,
            "proposed_content": self.proposed_content,
            "related_skill": self.related_skill,
        }


@dataclass
class CuratorReviewResult:
    findings: list[CuratorFinding] = field(default_factory=list)
    message: str = ""

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)


def _new_finding_id() -> str:
    return uuid.uuid4().hex[:12]


def _parse_date(value: str) -> date | None:
    if not value or not value.strip():
        return None
    text = value.strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def analyze_stale_skills(
    active_skills: list[dict[str, Any]],
    stale_days: int,
) -> list[CuratorFinding]:
    """Flag active skills not used within stale_days (or never used)."""
    if stale_days <= 0:
        return []

    cutoff = date.today() - timedelta(days=stale_days)
    findings: list[CuratorFinding] = []

    for meta in active_skills:
        name = str(meta.get("name", ""))
        if not name:
            continue
        last_used = str(meta.get("last_used", ""))
        parsed = _parse_date(last_used)
        if parsed is None:
            findings.append(
                CuratorFinding(
                    finding_id=_new_finding_id(),
                    finding_type=FINDING_STALE,
                    skill_name=name,
                    rationale=(
                        f"Skill '{name}' has no last_used date and may be unused."
                    ),
                    proposed_action=ACTION_ARCHIVE,
                )
            )
        elif parsed < cutoff:
            findings.append(
                CuratorFinding(
                    finding_id=_new_finding_id(),
                    finding_type=FINDING_STALE,
                    skill_name=name,
                    rationale=(
                        f"Skill '{name}' was last used on {last_used} "
                        f"(older than {stale_days} days)."
                    ),
                    proposed_action=ACTION_ARCHIVE,
                )
            )
    return findings


def analyze_bloated_skills(
    skill_manager: SkillManager,
    active_skills: list[dict[str, Any]],
    max_chars: int,
) -> list[CuratorFinding]:
    """Flag skills whose file content exceeds max_chars."""
    if max_chars <= 0:
        return []

    findings: list[CuratorFinding] = []
    for meta in active_skills:
        name = str(meta.get("name", ""))
        if not name:
            continue
        content = skill_manager.get_skill_content(name)
        if not content:
            continue
        if len(content) > max_chars:
            findings.append(
                CuratorFinding(
                    finding_id=_new_finding_id(),
                    finding_type=FINDING_BLOAT,
                    skill_name=name,
                    rationale=(
                        f"Skill '{name}' is {len(content)} chars "
                        f"(limit {max_chars}). Consider compacting."
                    ),
                    proposed_action=ACTION_COMPACT,
                    proposed_content=content,
                )
            )
    return findings


def _run_curator_completion(
    runtime: ModelRuntime,
    messages: list[dict[str, str]],
) -> str:
    response = runtime.create_chat_completion(
        messages,
        stream=False,
        **CURATOR_COMPLETION_OVERRIDES,
    )
    return response["choices"][0]["message"]["content"].strip()


def _parse_json_response(raw: str) -> dict[str, Any] | list[Any] | None:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def detect_duplicates(
    runtime: ModelRuntime,
    skill_manager: SkillManager,
    active_skills: list[dict[str, Any]],
) -> list[CuratorFinding]:
    """Use LLM to find overlapping or duplicate active skills."""
    if len(active_skills) < 2:
        return []

    summaries: list[str] = []
    for meta in active_skills:
        name = str(meta.get("name", ""))
        desc = str(meta.get("description", ""))
        content = skill_manager.get_skill_content(name) or ""
        trigger = ""
        if "## Trigger" in content:
            parts = content.split("## Trigger", 1)[-1].split("##", 1)[0]
            trigger = parts.strip()[:200]
        summaries.append(f"- {name}: {desc}. Trigger: {trigger or '(none)'}")

    system_prompt = (
        "You are a skill curator. Review active skills and find duplicates or "
        "near-duplicates (same workflow, overlapping purpose).\n\n"
        "Active skills:\n"
        f"{chr(10).join(summaries)}\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"duplicates": [{"keep": "skill_to_keep", "archive": "skill_to_archive", '
        '"rationale": "why they overlap"}]}\n\n'
        "Rules:\n"
        "- Only list pairs that are genuinely redundant.\n"
        "- keep = the better-named or more complete skill; archive = the redundant one.\n"
        "- If no duplicates: {\"duplicates\": []}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Find duplicate skill pairs. JSON only."},
    ]

    try:
        raw = _run_curator_completion(runtime, messages)
    except Exception as error:  # noqa: BLE001
        print(f"[curator] Duplicate detection failed: {error}")
        return []

    data = _parse_json_response(raw)
    if not isinstance(data, dict):
        return []

    pairs = data.get("duplicates", [])
    if not isinstance(pairs, list):
        return []

    active_names = {str(m.get("name", "")) for m in active_skills}
    findings: list[CuratorFinding] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        keep = str(pair.get("keep", "")).strip()
        archive = str(pair.get("archive", "")).strip()
        rationale = str(pair.get("rationale", "Duplicate or overlapping skill.")).strip()
        if not archive or archive not in active_names:
            continue
        if keep and keep not in active_names:
            continue
        findings.append(
            CuratorFinding(
                finding_id=_new_finding_id(),
                finding_type=FINDING_DUPLICATE,
                skill_name=archive,
                rationale=rationale,
                proposed_action=ACTION_ARCHIVE,
                related_skill=keep,
            )
        )
    return findings


def generate_compaction(
    runtime: ModelRuntime,
    skill_content: str,
    max_chars: int,
) -> str:
    """Ask LLM to shorten skill markdown while preserving SKILL-001 sections."""
    system_prompt = (
        f"Compress the following skill markdown to at most {max_chars} characters. "
        "Preserve frontmatter and all three sections: ## Trigger, ## Procedure, "
        "## Validation. Keep all essential steps. Output only the compressed markdown."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": skill_content},
    ]
    try:
        raw = _run_curator_completion(runtime, messages)
    except Exception as error:  # noqa: BLE001
        print(f"[curator] Compaction failed: {error}")
        return skill_content

    compacted = raw.strip()
    if compacted.startswith("```"):
        fence = re.search(r"```(?:markdown)?\s*(.*?)\s*```", compacted, re.DOTALL)
        if fence:
            compacted = fence.group(1).strip()

    if validate_skill_content(compacted):
        return skill_content
    return compacted


def enrich_bloat_findings(
    runtime: ModelRuntime,
    findings: list[CuratorFinding],
    max_chars: int,
) -> list[CuratorFinding]:
    """Generate compacted content for bloat findings."""
    enriched: list[CuratorFinding] = []
    for finding in findings:
        if finding.finding_type != FINDING_BLOAT or not finding.proposed_content:
            enriched.append(finding)
            continue
        compacted = generate_compaction(
            runtime,
            finding.proposed_content,
            max_chars,
        )
        enriched.append(
            CuratorFinding(
                finding_id=finding.finding_id,
                finding_type=finding.finding_type,
                skill_name=finding.skill_name,
                rationale=finding.rationale,
                proposed_action=finding.proposed_action,
                proposed_content=compacted,
                related_skill=finding.related_skill,
            )
        )
    return enriched


def run_full_review(
    runtime: ModelRuntime,
    config: AppConfig,
    skill_manager: SkillManager,
    *,
    use_llm: bool = True,
) -> CuratorReviewResult:
    """Run stale, bloat, and duplicate analysis on active skills."""
    active = skill_manager.list_skills(status="active")
    if not active:
        return CuratorReviewResult(
            findings=[],
            message="No active skills to review.",
        )

    curator_cfg = config.curator
    findings: list[CuratorFinding] = []
    findings.extend(analyze_stale_skills(active, curator_cfg.stale_days))
    findings.extend(
        analyze_bloated_skills(
            skill_manager,
            active,
            curator_cfg.bloat_max_chars,
        )
    )

    if use_llm:
        bloat = [f for f in findings if f.finding_type == FINDING_BLOAT]
        other = [f for f in findings if f.finding_type != FINDING_BLOAT]
        findings = other + enrich_bloat_findings(
            runtime,
            bloat,
            curator_cfg.bloat_max_chars,
        )
        findings.extend(detect_duplicates(runtime, skill_manager, active))

    if not findings:
        return CuratorReviewResult(
            findings=[],
            message="Curator review complete: no issues found.",
        )

    return CuratorReviewResult(
        findings=findings,
        message=f"Curator found {len(findings)} suggestion(s).",
    )


def format_finding_view(finding: CuratorFinding, index: int, total: int) -> str:
    """Format a single finding for display."""
    lines = [
        f"Curator finding ({index + 1}/{total})",
        f"Type: {finding.finding_type}",
        f"Skill: {finding.skill_name}",
        f"Action: {finding.proposed_action}",
    ]
    if finding.related_skill:
        lines.append(f"Related: {finding.related_skill}")
    lines.extend(["", "Rationale:", finding.rationale])
    if finding.proposed_content and finding.proposed_action == ACTION_COMPACT:
        lines.extend(
            [
                "",
                f"Proposed compact content ({len(finding.proposed_content)} chars):",
                finding.proposed_content[:2000],
            ]
        )
        if len(finding.proposed_content) > 2000:
            lines.append("... (truncated)")
    return "\n".join(lines)


def format_review_view(findings: list[CuratorFinding]) -> str:
    """Format all pending findings for CLI display."""
    if not findings:
        return "No pending curator findings."
    parts = [f"Curator review: {len(findings)} finding(s)", ""]
    for index, finding in enumerate(findings):
        parts.append(format_finding_view(finding, index, len(findings)))
        parts.append("")
    parts.append("Use /curator in TUI or approve findings individually.")
    return "\n".join(parts).rstrip()
