"""__main__.py - Allow ``python -m scryer`` entrypoint execution."""

from __future__ import annotations

import sys
from scryer.cli import main

if __name__ == "__main__":
    sys.exit(main())

