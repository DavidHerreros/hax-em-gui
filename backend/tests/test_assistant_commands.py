"""The user's "/" chat commands: one text file each, CRUD over the API."""

from __future__ import annotations

import pytest

from hax_gui import assistant_commands


@pytest.fixture(autouse=True)
def commands_dir(tmp_path, monkeypatch):
    d = tmp_path / "assistant_commands"
    monkeypatch.setattr(assistant_commands, "commands_dir", lambda: d)
    return d


def test_round_trip(client, run, commands_dir):
    body = {"description": "Is the run healthy?", "prompt": "Look at {run}.\n\n---\nThen: {text}",
            "attach_run": True, "include_tab": True, "may_act": False, "pinned": True}
    r = run(client.put("/api/assistant/commands/check-run", json=body))
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "check-run", **body}
    assert (commands_dir / "check-run.md").read_text().startswith("---\ndescription: Is the run healthy?\n")
    assert run(client.get("/api/assistant/commands")).json() == [{"name": "check-run", **body}]

    assert run(client.delete("/api/assistant/commands/check-run")).status_code == 200
    assert run(client.get("/api/assistant/commands")).json() == []
    assert run(client.delete("/api/assistant/commands/check-run")).status_code == 404


def test_defaults_answer_only(client, run):
    r = run(client.put("/api/assistant/commands/next-step", json={"prompt": "What next?"}))
    assert r.status_code == 200
    assert r.json()["may_act"] is False and r.json()["pinned"] is False


@pytest.mark.parametrize("name", ["Bad", "-x", "a b", "x" * 33, "autonomy", "float"])
def test_rejects_bad_or_builtin_names(client, run, name):
    r = run(client.put(f"/api/assistant/commands/{name}", json={"prompt": "p"}))
    assert r.status_code == 400


def test_rejects_empty_prompt_and_traversal(client, run, commands_dir):
    assert run(client.put("/api/assistant/commands/empty", json={"prompt": "  "})).status_code == 400
    assert run(client.delete("/api/assistant/commands/..%2Fx")).status_code in (400, 404, 405)
    with pytest.raises(ValueError):
        assistant_commands.delete_command("../x")
    assert not commands_dir.exists()


def test_hand_written_file_without_header(commands_dir):
    commands_dir.mkdir()
    (commands_dir / "plain.md").write_text("Just a prompt.\n")
    (commands_dir / "Not A Command.md").write_text("ignored")
    assert assistant_commands.list_commands() == [
        {"name": "plain", "description": "", "prompt": "Just a prompt.", **assistant_commands.FLAGS}]
