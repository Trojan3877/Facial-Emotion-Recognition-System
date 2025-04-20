# Save training history to visualize later
import json

history_dict = history.history
with open("history.json", "w") as f:
    json.dump(history_dict, f)

print("📈 Training history saved to 'history.json'")
