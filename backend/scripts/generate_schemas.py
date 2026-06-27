#!/usr/bin/env python
"""Regenerate the cached program form-schemas.

Run inside the hax conda env (imports JAX). The backend serves the cached JSON,
so re-run this whenever a hax program's CLI changes.

    python scripts/generate_schemas.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hax_gui.config import settings  # noqa: E402
from hax_gui.introspect import extract_all  # noqa: E402


def main() -> int:
    result = extract_all()
    settings.schema_cache.parent.mkdir(parents=True, exist_ok=True)
    settings.schema_cache.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {len(result['programs'])} program schemas -> {settings.schema_cache}")
    if result["errors"]:
        print("Programs that failed to introspect:")
        for name, err in result["errors"].items():
            print(f"  - {name}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
