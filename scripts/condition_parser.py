# scripts/condition_parser.py
import re

# Extract conditions from user input
def extract_conditions(text):
    conditions = []
    
    # Example regex for "greater than", "less than", "between"
    greater_than_pattern = r"greater than (\d+)"
    less_than_pattern = r"less than (\d+)"
    between_pattern = r"between (\w+) and (\w+)"
    
    greater_than = re.findall(greater_than_pattern, text)
    less_than = re.findall(less_than_pattern, text)
    between = re.findall(between_pattern, text)
    
    if greater_than:
        conditions.append((">", int(greater_than[0])))
    if less_than:
        conditions.append(("<", int(less_than[0])))
    if between:
        conditions.append(("BETWEEN", between[0][0], between[0][1]))
    
    return conditions
