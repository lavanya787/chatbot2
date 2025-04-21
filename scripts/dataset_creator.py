import json
import os

LOG_FILE = "logs/query_logs.jsonl"
INTENT_OUT = "training_data/intent_classification.json"
ENTITY_OUT = "training_data/entity_dataset.json"

def create_datasets_from_logs():
    if not os.path.exists(LOG_FILE):
        print("No log file found.")
        return

    os.makedirs("training_data", exist_ok=True)

    intent_data = []
    entity_data = []

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                question = entry.get("question")
                intent = entry.get("intent")
                entities = entry.get("entities", {})

                if question and intent:
                    # Intent classification data
                    intent_data.append({
                        "text": question,
                        "label": intent
                    })

                    # Entity recognition data
                    # This format works well for spaCy / NER training
                    entity_list = []
                    for ent_type, ent_values in entities.items():
                        for val in ent_values:
                            start = question.lower().find(val.lower())
                            if start != -1:
                                entity_list.append({
                                    "start": start,
                                    "end": start + len(val),
                                    "label": ent_type.upper()
                                })

                    entity_data.append({
                        "text": question,
                        "entities": entity_list
                    })

            except json.JSONDecodeError:
                print("Skipping corrupted line in log.")

    # Save outputs
    with open(INTENT_OUT, 'w', encoding='utf-8') as f:
        json.dump(intent_data, f, indent=2)

    with open(ENTITY_OUT, 'w', encoding='utf-8') as f:
        json.dump(entity_data, f, indent=2)

    print(f"✅ Dataset created:\n - {INTENT_OUT}\n - {ENTITY_OUT}")
