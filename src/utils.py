import json
# import re
import regex as re
from pathlib import Path


def read_json(filename: Path) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data


def write_json(data: dict, filename: Path) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_document(filename: Path) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
        return text


def extract_json(text: str) -> dict | None:
    # Regex to capture JSON object from messy text
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            # Optional: repair JSON using a library like json-repair
            pass
    return None


def check_json(make_json: dict, prompt: str) -> str | bool:
    retry_prompt = f"""Your previous response was invalid. The output JSON.
            must have the following structure: 

            {{
              "defects": [
                {{
                  "label": "defect label you identified or null if not identified",
                  "severity": "severity of the defect or null if not identified",
                  "location": "location of the vessel you spotted the defect or null if not identified",
                  "description": "short description of the defect or null if not identified"
                }}
              ]
            }} 

            Here is the original task:

            {prompt}
            """

    if 'defects' not in make_json:
        return retry_prompt

    defects = make_json['defects']
    for item in defects:

        if 'label' not in item or 'severity' not in item or 'location' not in item or 'description' not in item:
            return retry_prompt

    return True
