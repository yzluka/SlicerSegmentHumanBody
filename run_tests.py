import sys, os, io, unittest

sys.path.insert(0, r'D:\SlicerSegmentHumanBody\SegmentHumanBody')
sys.path.insert(0, r'D:\SlicerSegmentHumanBody\SegmentHumanBody\Testing\Python')

# Slicer's bundled Python on Windows defaults stdout to cp1252, which cannot
# encode the → characters used in test docstrings.  Force UTF-8 so the test
# runner can write all descriptions without crashing.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Allow the Slicer GUI (layout manager, slice widgets) to finish initializing
# before the test module is imported.  Without this, --python-script mode may
# load the module before slicer.app.layoutManager() becomes non-None.
try:
    import slicer
    for _ in range(10):
        slicer.app.processEvents()
except Exception:
    pass

import SegmentHumanBodyTest as ext

suite = unittest.TestLoader().loadTestsFromModule(ext)
result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
print(f'\n{"OK" if result.wasSuccessful() else "FAILED"} — {result.testsRun} tests, '
      f'{len(result.errors)} errors, {len(result.failures)} failures', flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
