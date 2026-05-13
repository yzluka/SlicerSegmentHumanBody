import json
import sys

sys.path.insert(0, r"D:\SlicerSegmentHumanBody\SegmentHumanBody")

from core.TimeLogSummarizer import TimeLogSummarizer


INPUT_PATH = r"C:\Users\82755\Desktop\time_recording.json"
JSON_OUTPUT_PATH = r"C:\Users\82755\Desktop\time_recording_summary.json"
TEXT_OUTPUT_PATH = r"C:\Users\82755\Desktop\time_recording_summary.txt"


with open(INPUT_PATH, encoding="utf-8") as f:
    semantic = json.load(f)

result = TimeLogSummarizer(semantic).export()

with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n\n".join(result["text"]))
    f.write("\n")

print(f"Written: {len(result['spans'])} spans")
print(JSON_OUTPUT_PATH)
print(TEXT_OUTPUT_PATH)
