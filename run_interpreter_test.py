import json, sys
sys.path.insert(0, r"D:\SlicerSegmentHumanBody\SegmentHumanBody")
from core.TimeLogInterpreter import TimeLogInterpreter

with open(r"C:\Users\82755\Desktop\time_recording_raw.json") as f:
    raw = json.load(f)

result = TimeLogInterpreter(raw).export()

with open(r"C:\Users\82755\Desktop\time_recording.json", "w") as f:
    json.dump(result, f, indent=2)

print("Written: {} events".format(len(result["events"])))
print()
for ev in result["events"]:
    eid = ev.get("id")
    ts = ev.get("timestamp", "")[-12:]
    ijk = ev.get("ijk")
    mouse = ev.get("mouse")
    kind = ev.get("kind", "")[:4]
    tool = ev.get("tool")
    evt_name = ev.get("event", "")
    extra = ""
    if ev.get("brush_mm"):
        extra = " brush={}mm".format(ev["brush_mm"])
    if evt_name:
        extra += " [{}]".format(evt_name)
    if ev.get("point_name"):
        extra += " pt={}".format(ev["point_name"])
    print("{:3}  {}  {!s:22}  {:7} {} {!s:6}{}".format(
        eid, ts, ijk, mouse, kind, tool, extra))
