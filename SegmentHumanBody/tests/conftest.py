"""
Add the SegmentHumanBody package directory to sys.path so that
`import core.*` resolves correctly when pytest is run from the repo root
or from within the SegmentHumanBody/ directory.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_path():
    """Override pytest's tmp_path to avoid Windows basetemp PermissionError."""
    d = Path(tempfile.mkdtemp(prefix='pytest_shb_'))
    yield d
    shutil.rmtree(d, ignore_errors=True)
