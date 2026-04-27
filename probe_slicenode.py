"""Probe: check what slice infrastructure exists in --no-main-window mode."""
import slicer, sys

lm = slicer.app.layoutManager()
print(f"layoutManager: {lm}")

for name in ("Red", "Green", "Yellow"):
    node_id = f"vtkMRMLSliceNode{name}"
    n = slicer.mrmlScene.GetNodeByID(node_id)
    print(f"{node_id}: {n}")

sys.exit(0)
