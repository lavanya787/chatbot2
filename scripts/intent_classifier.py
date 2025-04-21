from transformers import pipeline
from rapidfuzz import process
import json

# Load fallback intents from JSON
def load_fallback_intents(file_path="scripts/fallback_intents.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load the transformer model for intent classification
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_intent(question, fallback_intents):
    # Use zero-shot classification to classify the intent
    candidate_labels = ["get_expenses", "get_material_usage", "get_total", "get_project_details"]
    result = intent_classifier(question, candidate_labels)

    # If the model's confidence is low, fall back to keyword matching
    if result["scores"][0] < 0.7:  # If confidence < 70%
        for keyword, intent in fallback_intents.items():
            if keyword in question.lower():
                return intent

    return result["labels"][0]  # Return the top predicted intent
