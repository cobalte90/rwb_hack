from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.loaders import get_runtime_context, profile_status


def main() -> None:
    context = get_runtime_context()
    print(profile_status(context))


if __name__ == "__main__":
    main()
