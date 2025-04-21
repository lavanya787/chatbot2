import spacy

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

def extract_entities(question):
    doc = nlp(question)
    categories = []
    dates = []
    prices = []

    for ent in doc.ents:
        if ent.label_ == "MONEY":
            prices.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
        else:
            categories.append(ent.text)

    return {"categories": categories, "dates": dates, "prices": prices}
