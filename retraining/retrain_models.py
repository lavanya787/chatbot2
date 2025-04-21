import json
import os
from datetime import datetime
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, TrainingArguments,
    Trainer, DataCollatorWithPadding, DataCollatorForTokenClassification,
    pipeline
)
import numpy as np

# Load paths from config
with open("config/paths.json") as f:
    config = json.load(f)

LOG_FILE = config["log_file"]
INTENT_MODEL = config["base_intent_model"]
NER_MODEL = config["base_ner_model"]
DATE_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
INTENT_DIR = os.path.join(config["intent_model_output_dir"], DATE_TAG)
NER_DIR = os.path.join(config["ner_model_output_dir"], DATE_TAG)

ENTITY_TAGS = ["O", "B-CATEGORY", "I-CATEGORY", "B-PROJECT", "I-PROJECT", "B-DATE", "I-DATE", "B-NUM", "I-NUM"]
label2id = {tag: i for i, tag in enumerate(ENTITY_TAGS)}
id2label = {i: tag for tag, i in label2id.items()}

# ------------------- INTENT ---------------------
def prepare_intent_data():
    with open(LOG_FILE) as f:
        logs = json.load(f)
    examples = [{"text": log["question"], "label": log["predicted_intent"]}
                for log in logs if log.get("predicted_intent") not in [None, "unknown_intent"]]
    labels = sorted(set(e["label"] for e in examples))
    label2id_map = {label: i for i, label in enumerate(labels)}
    for e in examples:
        e["label"] = label2id_map[e["label"]]
    return examples, label2id_map

def train_intent():
    examples, label2id_map = prepare_intent_data()
    train, test = train_test_split(examples, test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "test": Dataset.from_list(test)
    }).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_MODEL, num_labels=len(label2id_map)
    )

    args = TrainingArguments(
        output_dir=INTENT_DIR, per_device_train_batch_size=8, num_train_epochs=3,
        evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        logging_dir=os.path.join(INTENT_DIR, "logs")
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    model.save_pretrained(INTENT_DIR)
    tokenizer.save_pretrained(INTENT_DIR)
    with open(os.path.join(INTENT_DIR, "label2id.json"), "w") as f:
        json.dump(label2id_map, f)
    print(f"✅ Intent model saved to {INTENT_DIR}")

# ------------------- ENTITY ---------------------
def prepare_ner_data():
    with open(LOG_FILE) as f:
        logs = json.load(f)

    data = []
    for log in logs:
        q = log.get("question")
        ents = log.get("entities", {})
        if not q or not ents:
            continue

        tokens = q.strip().split()
        labels = ["O"] * len(tokens)

        for ent_type, values in ents.items():
            if not isinstance(values, list):
                values = [values]
            for val in values:
                val_tokens = val.strip().split()
                for i in range(len(tokens) - len(val_tokens) + 1):
                    if tokens[i:i+len(val_tokens)] == val_tokens:
                        labels[i] = f"B-{ent_type.upper()}"
                        for j in range(1, len(val_tokens)):
                            labels[i+j] = f"I-{ent_type.upper()}"
                        break
        data.append({"tokens": tokens, "labels": labels})
    return data

def tokenize_ner_batch(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], 0))
            else:
                label_ids.append(label2id.get(label[word_idx], 0))
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

def train_ner():
    data = prepare_ner_data()
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
    train_data, test_data = train_test_split(data, test_size=0.2)

    ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    }).map(lambda x: tokenize_ner_batch(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=NER_DIR, per_device_train_batch_size=8, num_train_epochs=3,
        evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        logging_dir=os.path.join(NER_DIR, "logs")
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    trainer.train()
    model.save_pretrained(NER_DIR)
    tokenizer.save_pretrained(NER_DIR)
    print(f"✅ NER model saved to {NER_DIR}")

# Save latest model paths
latest_config = {
    "intent_model": INTENT_DIR,
    "ner_model": NER_DIR
}
with open("config/latest_models.json", "w") as f:
    json.dump(latest_config, f, indent=2)
print("📝 Updated latest_models.json")

# ------------------- MAIN ---------------------
if __name__ == "__main__":
    print("🔁 Starting intent retraining...")
    train_intent()

    print("🔁 Starting entity retraining...")
    train_ner()

    print("🎉 All models retrained successfully.")
