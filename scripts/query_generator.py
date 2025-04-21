# scripts/query_generator.py
import json

def generate_query(intent, entities):
    category = entities["categories"][0] if entities["categories"] else None
    date = entities["dates"][0] if entities["dates"] else None

    if intent == "get_total":
        return f"SELECT SUM(amount) FROM sales_data WHERE product_category = '{category}'"

    elif intent == "get_average":
        return f"SELECT AVG(amount) FROM sales_data WHERE product_category = '{category}'"

    elif intent == "get_count":
        return f"SELECT COUNT(*) FROM sales_data WHERE product_category = '{category}'"

    elif intent == "get_max":
        return f"SELECT MAX(amount) FROM sales_data WHERE product_category = '{category}'"

    elif intent == "get_min":
        return f"SELECT MIN(amount) FROM sales_data WHERE product_category = '{category}'"

    elif intent == "compare":
        return f"SELECT product_category, SUM(amount) FROM sales_data GROUP BY product_category"

    elif intent == "filter_data":
        query = "SELECT * FROM sales_data"
        conditions = []
        if category:
            conditions.append(f"product_category = '{category}'")
        if date:
            conditions.append(f"transaction_date = '{date}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        return query

    else:
        raise ValueError(f"Unknown intent: {intent}")
