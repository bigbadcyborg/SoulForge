"""Basic tests for the Skill System (Iteration 7)."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from app.skills.skill_manager import SkillManager
from app.core.config import AppConfig

def test_skill_manager_basic(tmp_path):
    # Setup mock config with tmp directories
    config = MagicMock()
    config.skills.active_path = str(tmp_path / "active")
    config.skills.archived_path = str(tmp_path / "archived")
    config.skills.registry_path = str(tmp_path / "registry.json")

    mgr = SkillManager(config)
    
    # Create skill
    success = mgr.create_skill("test_skill", "A test skill", "Procedure: step 1", ["pytest"])
    assert success is True
    assert (tmp_path / "active" / "test_skill.md").exists()
    
    # List skills
    skills = mgr.list_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "test_skill"
    
    # Get content
    content = mgr.get_skill_content("test_skill")
    assert "Procedure: step 1" in content
    
    # Archive skill
    mgr.archive_skill("test_skill")
    assert not (tmp_path / "active" / "test_skill.md").exists()
    assert (tmp_path / "archived" / "test_skill.md").exists()
    
    skills = mgr.list_skills(status="active")
    assert len(skills) == 0
    archived = mgr.list_skills(status="archived")
    assert len(archived) == 1
    
    # Delete skill
    mgr.delete_skill("test_skill")
    assert not (tmp_path / "archived" / "test_skill.md").exists()
    assert len(mgr.list_skills(status="archived")) == 0

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
