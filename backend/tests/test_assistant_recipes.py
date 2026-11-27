"""The user's recipe in what the GUI shows: the workflow card's flags and reasons, the draft job card,
and the restart the monitor proposes for a run that died after resuming from its checkpoint."""

from __future__ import annotations

import json

import pytest

from test_assistant_workflows import ask, data, isolated, schema  # noqa: F401  (fixtures)

from hax_gui import assistant as assistant_bridge
from hax_gui import assistant_actions, assistant_workflows

pytestmark = pytest.mark.skipif(not assistant_bridge.available(), reason="hax_assistant is not installed")
RESUMED = ("\x1b[93m\nCheckpoint detected: resuming training from epoch 5\x1b[0m\nTraceback (most recent call last):\n"
           "UnboundLocalError: cannot access local variable 'occ_alpha' where it is not associated with a value\n")


def test_workflow_card_starts_from_the_recipe(project, data):
    out = assistant_workflows.preview(ask(data), project["id"])
    assert out["ok"], (out["errors"], [c for c in out["plan"]["checks"] if c["status"] == "fail"])
    train, predict = out["steps"][0], out["steps"][1]
    assert train["args"]["encoder_arch"] == "mlpnn" and train["args"]["batch_size"] == 32
    assert train["reasons"]["encoder_arch"] == "from your recipe hetsiren motions (hax default: convnn)"
    assert "encoder_arch" not in predict["args"] and "film_heads" not in predict["args"]
    assert any(c["status"] == "info" and "hetsiren motions" in c["message"] for c in out["plan"]["checks"])
    assert "Recipe: hetsiren motions" in out["plan"]["describe"]


def test_draft_job_starts_from_the_recipe(data):
    spec = assistant_actions.build_draft_spec("hetsiren", str(data["md"]))
    assert spec["recipe"] == "hetsiren motions"
    assert spec["args"]["encoder_arch"] == "mlpnn" and spec["args"]["batch_size"] == 32 and spec["args"]["epochs"] == 150
    assert spec["args"]["md"] == str(data["md"]) and "motion_only" not in spec["args"]
    from hax_assistant import recipes
    recipes.set_enabled("hetsiren motions", False)
    spec = assistant_actions.build_draft_spec("hetsiren", str(data["md"]))
    assert spec["recipe"] is None and spec["args"].get("encoder_arch") in (None, "convnn")


def test_a_crash_after_resume_is_restarted_from_scratch(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "HetSIREN_metrics").mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({"output_path": str(run_dir), "batch_size": 32}))
    before = assistant_bridge.watch_report(str(run_dir), pid_alive=False)["proposal"]
    assert before["kind"] == "crashed" and before["restart"] == "resume"
    (run_dir / "run.log").write_text("stub: hetsiren\n" + RESUMED)          # the GUI job's log
    after = assistant_bridge.watch_report(str(run_dir), pid_alive=False)["proposal"]
    assert after["restart"] == "scratch" and after["changes"] == {} and "occ_alpha" in after["details"][0]
    assert assistant_bridge.remedy_for(str(run_dir), "crashed", pid_alive=False)["restart"] == "scratch"
