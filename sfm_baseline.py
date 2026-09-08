"""Backward-compatible facade for the modular classical SfM package.

Existing callers may continue to use ``import sfm_baseline as sfm``. New code
can import the same public functions directly from :mod:`sfm`.
"""

from sfm import *  # noqa: F401,F403 - compatibility re-export by design
from sfm.cli import main
from sfm.visualization import _render_worker


if __name__ == "__main__":
    main()
