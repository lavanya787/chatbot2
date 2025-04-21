import json
from rapidfuzz import fuzz, process

# Load the column mapping dictionary from the JSON file
def column_mapping(file_path="scripts/column_mapping.json"):
    try:
        with open(file_path, 'r') as f:
            column_mapping_dict = json.load(f)
        return column_mapping_dict
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}

def map_column(user_input):
    # Try entire sentence first
    result = process.extractOne(user_input, column_mapping.keys(), scorer=fuzz.partial_ratio)
    if result:
        best_match, score, _ = result
        if score >= 75:
            return column_mapping[best_match]

    # Then fall back to each extracted category
    for word in user_input.split():
        result = process.extractOne(word, column_mapping.keys(), scorer=fuzz.ratio)
        if result:
            best_match, score, _ = result
            if score >= 85:
                return column_mapping[best_match]
    return None

# Main function to handle user questions and process the intent and entities
def main():
    column_mapping_dict = column_mapping()  # Load column mappings from JSON file

    while True:
        question = input("Ask a question (type 'exit' or 'quit' to exit): ")

        # Exit condition
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Predict intent and extract entities (you'll have your own NLP logic here)
        # For this example, we simulate it with a basic structure.
        predicted_intent = "get_average"  # This is just a placeholder; you should integrate your intent classifier
        entities = {'dates': [], 'categories': ['Physics'], 'prices': []}

        # Process categories to map them to column names
        if entities['categories']:
            mapped_column = map_column(entities['categories'][0], column_mapping_dict)  # Pass both arguments
            print(f"Predicted Intent: {predicted_intent}")
            print(f"Extracted Entities: {entities}")
            print(f"Mapped Column: {mapped_column}")
        else:
            print("No categories found to map.")

if __name__ == "__main__":
    main()
# This script is designed to load a column mapping from a JSON file and perform fuzzy matching on user input to find the best match in the mapping. It also includes a main loop for user interaction.
# The script uses the RapidFuzz library for fuzzy string matching, which is efficient and suitable for this task.               
