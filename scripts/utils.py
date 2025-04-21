# scripts/column_mapping.py
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Example column mapping lookup table
column_mapping = {
    "maths": "Maths",
    "physics": "Physics",
    "rent": "Rent",
    "total": "sum_of_amount",
}

# Find the closest matching column
def map_column(user_input):
    match = process.extractOne(user_input, column_mapping.keys())
    return column_mapping.get(match[0], None)
