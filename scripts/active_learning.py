# scripts/active_learning.py
import pickle
from scripts.intent_classifier import load_model

def retrain_model(new_data):
    # Load the current model
    model, vectorizer = load_model()

    # Train the model with new data (new_data is expected to be a list of (X_train, y_train) tuples)
    X_train, y_train = new_data
    X_train_vec = vectorizer.transform(X_train)
    model.fit(X_train_vec, y_train)

    # Save the retrained model
    with open('intent_model.pkl', 'wb') as model_file:
        pickle.dump((model, vectorizer), model_file)

    print("Model retrained successfully!")
