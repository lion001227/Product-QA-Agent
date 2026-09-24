import json

with open("summary.json","r",encoding="utf-8") as f:
    summary = json.load(f)

with open("results.json","r",encoding="utf-8") as f:
    results = json.load(f)

print(f"summary:",summary)
print(f"results:",results)
