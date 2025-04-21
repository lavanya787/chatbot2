import json
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import os

# === Load and encode data ===
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = label_encoder.transform(labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    labels = [d["intent"] for d in data]
    return texts, labels

# === Paths ===
data_path = "data/intent_data.json"
model_save_path = "model"

# === Load data ===
texts, labels = load_data(data_path)
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# === Save label list for inference ===
os.makedirs(model_save_path, exist_ok=True)
with open(os.path.join(model_save_path, "label_list.json"), "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

# === Tokenize ===
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
dataset = IntentDataset(texts, labels, tokenizer, label_encoder)

# === Model ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# === Trainer ===
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="no",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# === Train and Save ===
trainer.train()
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("✅ Model training complete and saved to /model/")
