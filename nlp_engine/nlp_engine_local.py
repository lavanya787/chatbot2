import nltk
import spacy
import json
import re
from datetime import datetime,  timezone
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
from fuzzywuzzy import process
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Force langdetect consistency
DetectorFactory.seed = 0

# Downloads
nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # default to English if detection fails
    
def translate_from_english(text, target_lang):
    if target_lang != 'en':
        try:
            return GoogleTranslator(source='en', target=target_lang).translate(text)
        except Exception as e:
            print(f"Translation failed: {e}")
            return text
    return text

# Pipelines
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants
INTENTS = [
    "predict", "classify", "analyze", "summarize", "compare", "explain",
    "retrieve", "filter", "visualize", "evaluate", "optimize", "debug", "correlate", "validate",
    "recommend", "generate", "forecast", "infer", "detect"
]

KEY_TARGET_WORDS = [
    "price", "score", "result", "grade", "output", "target", "label", "prediction", "forecast",
    "salary", "value", "accuracy", "performance", "trend", "relation", "insight", "probability"
]

OUTPUT_FORMATS = ["value", "label", "probability", "number", "range", "percentage", "score", "summary", "chart"]

GREETINGS = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}

CONDITION_KEYWORDS = {
    'if', 'where', 'when', 'greater', 'less', 'more than', 'equal', 'above', 'below', 'between',
    'before', 'after', 'during', 'until', 'with', 'without'
}

# Check greeting
def is_greeting(text):
    return text.lower().strip() in GREETINGS

# Classify intent
def extract_intent(question):
    try:
        result = zero_shot_classifier(question, INTENTS)
        return result['labels'][0]
    except Exception:
        return "analyze"

# Extract features and target
def extract_features_and_target(text):
    doc = nlp(text)
    features, target = [], None
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(word in chunk_text for word in KEY_TARGET_WORDS):
            target = chunk.text
        else:
            features.append(chunk.text)
    return list(set(features)), target

# Extract named entities
def extract_named_entities(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents]))

# Extract condition-related words
def extract_conditions(text):
    tokens = nltk.word_tokenize(text.lower())
    return list(set([w for w in tokens if w in CONDITION_KEYWORDS])) or None

# Determine output format
def extract_output_format(text):
    for fmt in OUTPUT_FORMATS:
        if fmt in text.lower():
            return fmt
    return "label"

# Sentiment
def extract_sentiment(text):
    try:
        result = sentiment_analyzer(text)
        return result[0]["label"]
    except:
        return "neutral"

# Log to file with timestamp
def log_question(question: str, corrected_question: str = None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = {"timestamp": timestamp, "question": question}
    if corrected_question and corrected_question != question:
        entry["corrected_question"] = corrected_question

    with open("questions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    with open("question_logs.txt", "a", encoding="utf-8") as f_txt:
        f_txt.write(f"\n[{timestamp}]\n")
        f_txt.write(f"Original Question: {question}\n")
        if corrected_question and corrected_question != question:
            f_txt.write(f"Corrected Question: {corrected_question}\n")

# Semantic similarity score (optional use)
def compute_similarity(q1, q2):
    emb1 = similarity_model.encode(q1, convert_to_tensor=True)
    emb2 = similarity_model.encode(q2, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2).item())

# Add this at the top of your file
previous_context = {
    "last_question": None,
    "last_features": None,
    "last_target": None,
    "last_intent": None
}

# Helper function to Detect follow-up type questions
def is_follow_up_question(q: str) -> bool:
    q = q.lower()
    follow_ups = [
        "compare it", "what about it", "how does it differ", "is it better", 
        "explain more", "why", "details", "same for this", "how", "tell me more"
    ]
    return any(phrase in q for phrase in follow_ups)

# Sample known lists (customize as needed)
known_features = [
    "accuracy", "precision", "recall", "f1 score", "speed", "latency",
    "efficiency", "scalability", "cost", "training time", "inference time",
    "robustness", "fairness", "bias", "explainability", "complexity"
]
known_targets = [
    "bert", "gpt", "transformer", "svm", "random forest", "naive bayes",
    "logistic regression", "cnn", "rnn", "lstm", "dataset", "model", "system", "algorithm"
]

# Spell correction
def correct_spelling(text: str) -> str:
    blob = TextBlob(text)
    return str(blob.correct())

# Fuzzy match for features and targets
def fuzzy_match(input_term, known_terms, threshold=80):
    match, score = process.extractOne(input_term, known_terms)
    return match if score >= threshold else input_term
# Function to process questions
def process_questions(questions):
    for question in questions:
        try:
            lang = detect(question)
            translated = GoogleTranslator(source='auto', target='en').translate(question)
            print("🔸 Original:", question)
            print("🌐 Detected Language:", lang)
            print("🔤 Translated to English:", translated)
            corrected = correct_spelling(translated)
            intent = extract_intent(corrected)
            features, target = extract_features_and_target(corrected)
            named_entities = extract_named_entities(corrected)
            conditions = extract_conditions(corrected)
            output_format = extract_output_format(corrected)
            sentiment = extract_sentiment(corrected)
            timestamp = datetime.now(timezone.utc).isoformat()

            # Log the question
            log_question(question, corrected)

            # Print structured info
            print("\nStructured input to send to model:")
            print("language:", lang)
            print("original_question:", question)
            print("translated_question:", corrected)
            print("intent:", intent)
            print("features:", features)
            print("target:", target)
            print("conditions:", conditions)
            print("named_entities:", named_entities)
            print("output_format:", output_format)
            print("sentiment:", sentiment)
            print("timestamp:", timestamp)
            print("-" * 80)

        except Exception as e:
            print(f"❌ Error processing question: {e}")
        # Main NLP Engine
def nlp_engine(question):
    # Step 1: Language Detection
    try:
        language = detect(question)
        if language not in ['en', 'es', 'fr', 'de', 'it']:  # acceptable list
            language = "es"  # fallback for this case
    except:
        language = "es"

    # Step 2: Translate if not English
    if language != "en":
        translated_question = GoogleTranslator(source='auto', target='en').translate(question)
    else:
        translated_question = question

    # Step 3: Intent Classification
    evaluation_keywords = ['performance', 'efficiency', 'accuracy', 'evaluate', 'precision', 'recall', 'f1']
    comparison_keywords = ['compare', 'difference between', 'vs', 'versus', 'better than']
    prediction_keywords = ['predict', 'forecast', 'estimate']

    lower_q = translated_question.lower()
    if any(word in lower_q for word in evaluation_keywords):
        intent = 'evaluate'
    elif any(word in lower_q for word in comparison_keywords):
        intent = 'compare'
    elif any(word in lower_q for word in prediction_keywords):
        intent = 'predict'
    else:
        intent = 'unknown'

    # Step 4: Extract features and target
    def extract_features_and_target(question_text):
        doc = nlp(question_text)
        features = [chunk.text.lower() for chunk in doc.noun_chunks 
                    if chunk.root.pos_ in ['NOUN', 'PROPN'] and chunk.text.lower() not in ['that', 'it', 'model']]
        features = list(set([f for f in features if len(f) > 2]))  # remove short/noisy terms

        # Target: look for words near "performance", "accuracy", etc.
        target_match = re.search(r"(performance|accuracy|result|efficiency|output)", question_text.lower())
        target = target_match.group(0) if target_match else None
        return features, target

    features, target = extract_features_and_target(translated_question)

    # Step 5: Output Format Detection
    output_format_match = re.search(r"(percentage|score|value|label)", translated_question.lower())
    output_format = output_format_match.group(0) if output_format_match else "score"

    # Step 6: Named Entities
    doc = nlp(translated_question)
    named_entities = [ent.text for ent in doc.ents]

    # Step 7: Sentiment Classification (simple rule-based)
    negative_words = ['bad', 'poor', 'low', 'negative']
    positive_words = ['good', 'great', 'high', 'excellent']
    sentiment = "neutral"
    if any(word in translated_question.lower() for word in negative_words):
        sentiment = "negative"
    elif any(word in translated_question.lower() for word in positive_words):
        sentiment = "positive"

    # Step 8: Final Output
    output = {
        "language": language,
        "original_question": question,
        "translated_question": translated_question,
        "intent": intent,
        "features": features,
        "target": target,
        "conditions": None,
        "named_entities": named_entities,
        "output_format": output_format,
        "sentiment": sentiment,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return output

# CLI with logging
if __name__ == "__main__":
    print("NLP Engine - Ask a question to analyze (type 'exit' to quit):")
    while True:
        q = input("👉 ")
        if q.lower() in ["exit", "quit"]:
            break

        output = nlp_engine(q)
        output["timestamp"] = datetime.now().isoformat()

        print("\nStructured input to send to model:")
        for k, v in output.items():
            print(f"{k}: {v}")

        # Append question log to file
        with open("questions.jsonl", "a") as f:
            f.write(json.dumps(output) + "\n")