import pytest

from mtplx.agent_profiles import AgentProfileStore
from mtplx.agent_workspace import WorkspaceConflictError, WorkspaceStoreError


def test_builtin_and_user_defined_profiles_are_durable(tmp_path):
    store = AgentProfileStore(tmp_path)
    builtins = {profile.id: profile for profile in store.list()}
    assert {"planner", "implementer", "reviewer", "tester", "research", "memory_curator"} <= set(builtins)
    assert builtins["implementer"].built_in is True
    assert {"write", "terminal"} <= set(builtins["implementer"].permissions)

    created = store.create(
        "docs-verifier",
        name="Docs verifier",
        description="Checks documentation examples.",
        permissions=["read", "search", "run_tests"],
        instructions="Run documentation examples and cite failures.",
        token_budget=4096,
        context_window=96_000,
        model="local-doc-model",
    )
    assert created.built_in is False
    assert created.sha256
    assert store.get(created.id).to_dict() == created.to_dict()

    updated = store.update(created.id, token_budget=5000, permissions=["read", "search"])
    assert updated.token_budget == 5000
    assert updated.permissions == ("read", "search")
    assert updated.sha256 != created.sha256


def test_profile_validation_protects_builtins_and_permissions(tmp_path):
    store = AgentProfileStore(tmp_path)
    with pytest.raises(WorkspaceConflictError, match="built-in"):
        store.create("reviewer", name="Replacement")
    with pytest.raises(WorkspaceStoreError, match="unknown agent permissions"):
        store.create("invalid", name="Invalid", permissions=["root-shell"])
    with pytest.raises(WorkspaceConflictError, match="built-in"):
        store.update("planner", instructions="replace")
