"""
Add the SegmentHumanBody package directory to sys.path so that
`import core.*` resolves correctly when pytest is run from the repo root
or from within the SegmentHumanBody/ directory.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
