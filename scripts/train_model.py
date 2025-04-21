# scripts/train_model.py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Sample data for training
X_train = ["What’s the total spend on food?", "Show students who failed in maths", 
           "What’s the average electricity bill?", "Did I spend more in March than April?"]
y_train = ['get_total', 'filter_data', 'get_average', 'compare']

# Vectorize the input questions
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save the trained model and vectorizer to a pickle file
with open('intent_model.pkl', 'wb') as model_file:
    pickle.dump((model, vectorizer), model_file)

print("Model training complete and saved!")
