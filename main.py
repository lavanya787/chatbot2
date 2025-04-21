import json
import spacy
import os
import datetime
from transformers import pipeline
from deep_translator import GoogleTranslator
from scripts.entity_extractor import extract_entities
from scripts.intent_classifier import classify_intent, load_fallback_intents
from scripts.query_generator import generate_query
from rapidfuzz import fuzz, process
#from scripts.email_notify import send_email

LOG_FILE_PATH = "logs/query_logs.jsonl"

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Define the intent classification model
INTENT_MODEL = "facebook/bart-large-mnli"  # Example model
classifier = pipeline("zero-shot-classification", model=INTENT_MODEL)

# Load fallback intent keywords
fallback_intents = load_fallback_intents('scripts/fallback_intents.json')

# Load column mapping from JSON
def load_column_mapping(file_path="scripts/column_mapping.json"):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
column_mapping = load_column_mapping()

# Map user input to a column using fuzzy matching
def map_column(user_input):
    result = process.extractOne(user_input, column_mapping.keys(), scorer=fuzz.ratio)
    if result:
        best_match, score, _ = result
        if score >= 80:
            return column_mapping[best_match]
    return None

# Function to translate input question into English using Google Translate
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)  # ✅ Correct

# Load the latest model paths
def load_latest_models():
    try:
        with open("config/latest_models.json") as f:
            latest = json.load(f)
            return latest["intent_model"], latest["ner_model"]
    except FileNotFoundError:
        print("Error: latest_models.json not found. Using default models.")
        return "facebook/bart-large-mnli", "en_core_web_sm"

INTENT_MODEL, NER_MODEL = load_latest_models()

# Transformer-based intent classifier with fallback
def classify_intent_with_transformer(question):
    classifier = pipeline("zero-shot-classification", model=INTENT_MODEL)
    candidate_labels = ["get_total", "get_average", "get_count", "get_max", "get_min", "compare", "filter_data"]

    result = classifier(question, candidate_labels)
    if result['scores'][0] > 0.8:
        return result['labels'][0]
    else:
        print("Low confidence in transformer model, falling back.")
        return fallback_intent_classification(question)

# Fallback keyword-based intent classifier
def fallback_intent_classification(question):
    fallback_intents = {
        "total sales": "get_total",
        "average price": "get_average",
        "best seller": "get_max"
    }
    for keyword, intent in fallback_intents.items():
        if keyword in question.lower():
            return intent
    return "unknown_intent"


# Intent classification using Zero-shot classification
def classify_intent(question):
    candidate_labels = ["get_total", "get_average", "get_count", "get_max", "get_min", "compare", "filter_data"]
    result = classifier(question, candidate_labels)
    if result['scores'][0] > 0.8:
        return result['labels'][0]
    else:
        return fallback_intent_classification(question)

# Entity extractor with spaCy + basic keyword fallback
def extract_entities_with_spacy(text):
    doc = nlp(text)
    entities = {
        "dates": [],
        "categories": [],
        "prices": []
    }

    fallback_categories = [
        'Dairy', 'Groceries', 'Milk', 'Fruits', 'Electronics',
        'Physics', 'Chemistry', 'Project A', 'Project B', 'Bricks',
        'Student', 'Patients', 'Names', 'Column', 'Sales'
    ]

    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["prices"].append(ent.text)
        elif ent.label_ in ["ORG", "PRODUCT", "GPE", "WORK_OF_ART"]:
            entities["categories"].append(ent.text)
    text_lower = text.lower()
    for category in fallback_categories:
        if category.lower() in text_lower and category not in entities['categories']:
            entities['categories'].append(category)

    return entities

def log_interaction(question, intent, entities, mapped_column, query):
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "intent": intent,
        "entities": entities,
        "mapped_column": mapped_column,
        "query": query
    }

    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")
        
def save_log(question, intent, entities, query=None):
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "question": question,
        "intent": intent,
        "entities": entities,
        "query": query
    }
    with open("query_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    with open("user_questions.txt", "a", encoding="utf-8") as qf:
        qf.write(question + "\n")


# Main loop
if __name__ == "__main__":
    print("Device set to use cpu")
    while True:
        question = input("Ask a question (type 'exit' or 'quit' to exit): ")

        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        #Translate the question into English
        translated_question = translate_to_english(question)
        print(f"Translated Question: {translated_question}")

        # Intent classification
        predicted_intent = classify_intent_with_transformer(translated_question)

        # Entity extraction
        entities = extract_entities_with_spacy(translated_question)

        # If no categories found, inform the user
        if not entities['categories'] and predicted_intent != "get_column":
            print("Error: No relevant categories found in your question.")
            save_log(question, predicted_intent, entities)
            continue

        mapped_column = None
        if predicted_intent != "get_column":
            mapped_column = map_column(entities['categories'][0]) if entities['categories'] else None
            if not mapped_column:
                print(f"No relevant column found for the extracted category: {entities['categories']}")
                save_log(question, predicted_intent, entities)
                continue
        else:
            mapped_column = "all_columns"

        try:
            query = generate_query(predicted_intent, entities)
        except Exception as e:
            print(f"Error generating query: {e}")
            save_log(question, predicted_intent, entities)
            continue

        # After generating the query and extracting necessary information
        output = {
            "language": "en",
            "original_question": question,
            "intent": predicted_intent,
            "features": entities['categories'] + entities['dates'] + entities['prices'],
            "target": None,
            "conditions": None,
            "named_entities": entities['dates'],
            "output_format": "score",
            "sentiment": "neutral",
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Print the structured output
        print(f"Language: {output['language']}")
        print(f"Original Question: {output['original_question']}")
        print(f"Predicted Intent: {output['intent']}")
        print(f"Features: {output['features']}")
        print(f"Target: {output['target']}")
        print(f"Conditions: {output['conditions']}")
        print(f"Named Entities: {output['named_entities']}")
        print(f"Output Format: {output['output_format']}")
        print(f"Sentiment: {output['sentiment']}")
        print(f"Timestamp: {output['timestamp']}")
        print(f"Predicted Intent: {predicted_intent}")
        print(f"Extracted Entities: {entities}")
        print(f"Mapped Column: {mapped_column}")
        print(f"Generated Query: {query}")
        
        save_log(f"{question} (translated: {translated_question})", predicted_intent, entities)
        
        # Step 7: Send Email Notification
        #body = (
         #   f"Question: {question}\n"
          #  f"Intent: {predicted_intent}\n"
           # f"Entities: {json.dumps(entities, indent=2)}\n"
            #f"Mapped Column: {mapped_column}\n"
            #f"Generated Query: {query}"
        #)
        #send_email("Query Processed", body)
