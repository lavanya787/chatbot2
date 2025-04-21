import schedule
import time
from scripts.intent_classifier import train_model
from scripts.active_learning import collect_feedback

def train_job():
    print("Training started.")
    train_model()

def feedback_collection_job():
    collect_feedback()

# Scheduling
schedule.every().day.at("03:00").do(train_job)  # Retrain at 3 AM
schedule.every().hour.do(feedback_collection_job)  # Collect feedback every hour

while True:
    schedule.run_pending()
    time.sleep(60)  # Wait a minute before running the next job
