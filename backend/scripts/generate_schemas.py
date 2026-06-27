#!/usr/bin/env python
"""Regenerate the cached program form-schemas.

Run inside the hax conda env (imports JAX). The backend serves the cached JSON,
so re-run this whenever a hax program's CLI changes.

    python scripts/generate_schemas.py          # -> writable user cache (data dir)
    python scripts/generate_schemas.py --seed    # -> packaged seed (commit before release)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hax_gui import genschemas  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate cached program form-schemas.")
    parser.add_argument("--seed", action="store_true",
                        help="write the schema cache bundled in the package (the shipped default), "
                             "instead of the per-user cache in the data dir")
    args = parser.parse_args()

    result = genschemas.regenerate(seed=args.seed)
    print(f"Wrote {len(result['programs'])} program schemas -> {result['_target']}")
    if result["errors"]:
        print("Programs that failed to introspect:")
        for name, err in result["errors"].items():
            print(f"  - {name}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
