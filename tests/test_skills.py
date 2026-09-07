from mtplx.skills import SkillStore


def test_skill_store_discovers_hash_and_portable_context(tmp_path):
    skill = tmp_path / ".mtplx" / "skills" / "reviewer"
    (skill / "references").mkdir(parents=True)
    (skill / "scripts").mkdir()
    (skill / "SKILL.md").write_text(
        "# Review changes\nCheck the diff and report evidence.\n",
        encoding="utf-8",
    )
    (skill / "references" / "checklist.md").write_text("check", encoding="utf-8")
    (skill / "scripts" / "check.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    store = SkillStore([tmp_path], user_root=tmp_path / "missing-user-skills")
    found = store.discover()
    assert len(found) == 1
    assert found[0].name == "reviewer"
    assert found[0].references
    assert found[0].scripts
    assert len(found[0].sha256) == 64
    assert "Review changes" in (store.context() or "")
    assert store.get("reviewer") == found[0]
