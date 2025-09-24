# v1.11
"""Top-level package for the BABYLLM phone integrations.

The project historically imported modules via a lower-case
``phone`` package path. However, the actual directory on disk is
upper-case (``PHONE``), which causes import failures on
case-sensitive filesystems. Making this directory a package and
registering a compatibility alias ensures that imports work
regardless of letter casing.
"""

from __future__ import annotations

import sys

# Provide an alias so ``import phone`` resolves to this package.
# This keeps backwards compatibility with previously shipped code
# that expected a lower-case package name.
sys.modules.setdefault("phone", sys.modules[__name__])

__all__ = []
