"""Smoke tests for iteration 9 curator."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.core.chat_controller import ChatController
from app.core.config import CuratorConfig, load_config
from app.skills.curator import (
    ACTION_ARCHIVE,
    ACTION_COMPACT,
    FINDING_BLOAT,
    FINDING_DUPLICATE,
    FINDING_STALE,
    CuratorFinding,
    analyze_bloated_skills,
    analyze_stale_skills,
    detect_duplicates,
    enrich_bloat_findings,
    run_full_review,
)
from app.skills.skill_manager import SkillManager


def test_analyze_stale_skills_by_date() -> None:
    old = (date.today() - timedelta(days=60)).isoformat()
    recent = date.today().isoformat()
    active = [
        {"name": "old_skill", "last_used": old},
        {"name": "fresh_skill", "last_used": recent},
        {"name": "never_used", "last_used": ""},
    ]
    findings = analyze_stale_skills(active, stale_days=30)
    names = {f.skill_name for f in findings}
    assert "old_skill" in names
    assert "never_used" in names
    assert "fresh_skill" not in names
    assert all(f.proposed_action == ACTION_ARCHIVE for f in findings)


def test_analyze_bloated_skills(tmp_path) -> None:
    config = MagicMock()
    config.skills.active_path = str(tmp_path / "active")
    config.skills.archived_path = str(tmp_path / "archived")
    config.skills.registry_path = str(tmp_path / "registry.json")

    mgr = SkillManager(config)
    long_body = "x" * 1200
    mgr.create_skill("big_skill", "big", long_body)

    active = [{"name": "big_skill", "description": "big"}]
    findings = analyze_bloated_skills(mgr, active, max_chars=1000)
    assert len(findings) == 1
    assert findings[0].finding_type == FINDING_BLOAT
    assert findings[0].proposed_action == ACTION_COMPACT


def test_detect_duplicates_parses_llm_response() -> None:
    runtime = MagicMock()
    runtime.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"duplicates": [{"keep": "skill_a", "archive": "skill_b", '
                        '"rationale": "Same CUDA rebuild workflow"}]}'
                    )
                }
            }
        ]
    }
    mgr = MagicMock()
    mgr.get_skill_content.return_value = "## Trigger\n\ntest\n\n## Procedure\n\n1. step\n\n## Validation\n\nok"
    active = [
        {"name": "skill_a", "description": "a"},
        {"name": "skill_b", "description": "b"},
    ]
    findings = detect_duplicates(runtime, mgr, active)
    assert len(findings) == 1
    assert findings[0].finding_type == FINDING_DUPLICATE
    assert findings[0].skill_name == "skill_b"
    assert findings[0].related_skill == "skill_a"


def test_enrich_bloat_findings() -> None:
    runtime = MagicMock()
    runtime.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        "---\nname: s\n---\n\n## Trigger\n\nt\n\n"
                        "## Procedure\n\n1. a\n\n## Validation\n\nv\n"
                    )
                }
            }
        ]
    }
    finding = CuratorFinding(
        finding_id="f1",
        finding_type=FINDING_BLOAT,
        skill_name="s",
        rationale="big",
        proposed_action=ACTION_COMPACT,
        proposed_content="x" * 2000,
    )
    enriched = enrich_bloat_findings(runtime, [finding], max_chars=1000)
    assert len(enriched) == 1
    assert "## Trigger" in enriched[0].proposed_content


def test_run_full_review_heuristics_only(tmp_path) -> None:
    config = load_config()
    config.curator = CuratorConfig(stale_days=30, bloat_max_chars=1000)
    config.skills.active_path = str(tmp_path / "active")
    config.skills.archived_path = str(tmp_path / "archived")
    config.skills.registry_path = str(tmp_path / "registry.json")
    config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")

    mgr = SkillManager(config)
    old = (date.today() - timedelta(days=90)).isoformat()
    mgr.create_skill("stale_one", "old skill", "short content")
    registry = mgr._load_registry()
    registry["stale_one"]["last_used"] = old
    mgr._save_registry(registry)

    runtime = MagicMock()
    result = run_full_review(runtime, config, mgr, use_llm=False)
    assert result.has_findings
    assert any(f.finding_type == FINDING_STALE for f in result.findings)


def test_controller_accept_archive_finding(tmp_path) -> None:
    controller = ChatController(load_config())
    controller.config.skills.active_path = str(tmp_path / "active")
    controller.config.skills.archived_path = str(tmp_path / "archived")
    controller.config.skills.registry_path = str(tmp_path / "registry.json")
    controller.config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")
    controller.skill_manager = SkillManager(controller.config)
    controller.set_feature("curator", True)
    controller.set_feature("skills", True)

    controller.skill_manager.create_skill("to_archive", "test", "body")
    finding = CuratorFinding(
        finding_id="abc123",
        finding_type=FINDING_STALE,
        skill_name="to_archive",
        rationale="stale",
        proposed_action=ACTION_ARCHIVE,
    )
    controller.pending_curator_findings = [finding]

    outcome = controller.accept_curator_finding("abc123")
    assert outcome.success
    assert controller.skill_manager.list_skills(status="active") == []
    assert len(controller.skill_manager.list_skills(status="archived")) == 1


def test_controller_archive_dismisses_other_findings_for_skill() -> None:
    controller = ChatController(load_config())
    controller.set_feature("curator", True)
    archive_finding = CuratorFinding(
        finding_id="arch1",
        finding_type=FINDING_STALE,
        skill_name="test_skill",
        rationale="stale",
        proposed_action=ACTION_ARCHIVE,
    )
    compact_finding = CuratorFinding(
        finding_id="cmp1",
        finding_type=FINDING_BLOAT,
        skill_name="test_skill",
        rationale="bloat",
        proposed_action=ACTION_COMPACT,
        proposed_content="compact body",
    )
    controller.pending_curator_findings = [archive_finding, compact_finding]
    controller._dismiss_findings_for_skill("test_skill")
    assert controller._visible_curator_findings() == []


def test_controller_compact_on_archived_skill_clears_finding(tmp_path) -> None:
    controller = ChatController(load_config())
    controller.config.skills.active_path = str(tmp_path / "active")
    controller.config.skills.archived_path = str(tmp_path / "archived")
    controller.config.skills.registry_path = str(tmp_path / "registry.json")
    controller.config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")
    controller.skill_manager = SkillManager(controller.config)
    controller.set_feature("curator", True)

    controller.skill_manager.create_skill("archived_skill", "test", "body")
    controller.skill_manager.archive_skill("archived_skill")

    finding = CuratorFinding(
        finding_id="cmp_arch",
        finding_type=FINDING_BLOAT,
        skill_name="archived_skill",
        rationale="bloat",
        proposed_action=ACTION_COMPACT,
        proposed_content="new content",
    )
    controller.pending_curator_findings = [finding]

    outcome = controller.accept_curator_finding("cmp_arch")
    assert outcome.success
    assert "archived" in outcome.message.lower() or "cleared" in outcome.message.lower()
    assert controller._visible_curator_findings() == []


def test_controller_dismiss_finding() -> None:
    controller = ChatController(load_config())
    finding = CuratorFinding(
        finding_id="dismiss1",
        finding_type=FINDING_STALE,
        skill_name="x",
        rationale="stale",
        proposed_action=ACTION_ARCHIVE,
    )
    controller.pending_curator_findings = [finding]
    controller.dismiss_curator_finding("dismiss1")
    assert controller._visible_curator_findings() == []


def test_controller_compact_direct(tmp_path) -> None:
    controller = ChatController(load_config())
    controller.config.skills.active_path = str(tmp_path / "active")
    controller.config.skills.archived_path = str(tmp_path / "archived")
    controller.config.skills.registry_path = str(tmp_path / "registry.json")
    controller.config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")
    controller.skill_manager = SkillManager(controller.config)
    controller.set_feature("curator", True)

    body = (
        "## Trigger\n\nwhen\n\n## Procedure\n\n" + ("step\n" * 200) + "\n## Validation\n\nok"
    )
    controller.skill_manager.create_skill("bloated", "big", body)

    compacted = (
        "---\nname: bloated\n---\n\n## Trigger\n\nt\n\n"
        "## Procedure\n\n1. s\n\n## Validation\n\nv\n"
    )
    with patch(
        "app.core.chat_controller.generate_compaction",
        return_value=compacted,
    ):
        result = controller.compact_skill_direct("bloated")
    assert result.has_findings
    assert controller.pending_curator_findings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
