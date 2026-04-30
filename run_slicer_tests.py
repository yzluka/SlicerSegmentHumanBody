"""Runner script for Slicer-native integration tests.

Usage (from Slicer's no-main-window mode):
  Slicer.exe --no-main-window --python-script D:/SlicerSegmentHumanBody/run_slicer_tests.py
"""

import sys
import io
import unittest

sys.path.insert(0, 'D:/SlicerSegmentHumanBody/SegmentHumanBody')

# Force UTF-8 on Windows where stdout defaults to cp1252.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import Testing.Python.SegmentHumanBodyTest as T

loader = unittest.TestLoader()
suite  = loader.loadTestsFromModule(T)
runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
