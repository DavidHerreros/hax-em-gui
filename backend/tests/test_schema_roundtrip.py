"""Every cached program schema must survive the schema -> form -> command trip.

Catches a schema cache that drifts from what command.build_command expects
(field types, flags, enums) without importing hax or JAX: required fields are
filled with type-appropriate dummies and the built argv must name the program
and every filled flag.
"""

from __future__ import annotations

import pytest


def _dummy(field):
    t = field.get("type")
    if t == "enum":
        return field["choices"][0]
    if t == "integer":
        return "4"
    if t == "number":
        return "1.5"
    if t == "boolean":
        return True
    return "/tmp/dummy"


def test_every_program_schema_builds_a_command(client):
    from hax_gui import registry
    from hax_gui.command import build_command

    programs = registry.list_programs()
    assert programs, "registry is empty"
    for p in programs:
        schema = registry.get_schema(p["name"])
        assert schema and schema.get("fields"), p["name"]
        values = {f["name"]: _dummy(f) for f in schema["fields"] if f.get("required")}
        argv = build_command(p["name"], values, gpu=None)
        assert p["name"] in argv, argv
        for f in schema["fields"]:
            if f.get("required") and f.get("type") != "boolean":
                assert f["flag"] in argv, (p["name"], f["flag"])


def test_strip_obsolete_drops_unknown_args(client):
    from hax_gui.command import strip_obsolete

    kept, dropped = strip_obsolete("estimate_latent_covariances",
                                   {"md": "/x", "argument_that_never_existed": "1"})
    assert "argument_that_never_existed" not in kept
    assert "argument_that_never_existed" in dropped
    assert kept.get("md") == "/x"


def test_output_arg_autofilled_not_required_from_user(client):
    from hax_gui import registry
    from hax_gui.command import ValidationError, build_command

    name = "estimate_latent_covariances"
    assert registry.output_arg(name) == "output_path"
    with pytest.raises(ValidationError):
        build_command(name, {"md": "/x"}, gpu=None)   # output_path is required...
    argv = build_command(name, {"md": "/x", "output_path": "/tmp/out"}, gpu=None)
    assert "--output_path" in argv                     # ...and satisfied when filled


def test_regenerate_keeps_previous_schema_when_introspection_fails(monkeypatch, tmp_path):
    """A transient import error (hax mid-edit, a dependency one step behind) must not
    empty the catalogue: the program keeps its last good schema and is reported under
    ``kept``, not ``errors``."""
    import json
    from hax_gui import genschemas, introspect
    from hax_gui.config import settings

    cache = tmp_path / "program_schemas.json"
    cache.write_text(json.dumps({"programs": {"hetsiren": {"name": "hetsiren", "fields": [{"name": "md"}]},
                                              "preprocess_particles": {"name": "preprocess_particles", "fields": []}},
                                 "errors": {}}))
    monkeypatch.setattr(type(settings), "schema_cache", property(lambda self: cache))
    monkeypatch.setattr(introspect, "extract_all", lambda: {
        "programs": {"preprocess_particles": {"name": "preprocess_particles", "fields": [{"name": "box"}]}},
        "errors": {"hetsiren": "ImportError: cannot import name 'tilt_rotation_matrices'",
                   "brand_new": "ImportError: no module"},
    })
    result = genschemas.regenerate()
    written = json.loads(cache.read_text())
    assert written["programs"]["hetsiren"]["fields"] == [{"name": "md"}]          # kept
    assert written["programs"]["preprocess_particles"]["fields"] == [{"name": "box"}]  # refreshed
    assert "hetsiren" in result["kept"] and "hetsiren" not in result["errors"]
    assert "brand_new" in result["errors"]                                        # nothing to keep
