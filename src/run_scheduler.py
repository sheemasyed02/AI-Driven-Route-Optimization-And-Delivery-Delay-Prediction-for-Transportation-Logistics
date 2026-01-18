import schedule
import time
from datetime import datetime
from mlops_automation import MLOpsManager

def weekly_retrain_job():
    print("\nWeekly Automated Retraining")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        mlops = MLOpsManager()
        mlops.check_and_retrain(performance_threshold=0.85)
        print("Weekly retraining completed successfully.")
    except Exception as e:
        print(f"Error during retraining: {e}")
        with open('logs/error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")

def main():
    print("MLOps Scheduler Started")
    print("Schedule: Weekly retraining every Monday at 2:00 AM")
    schedule.every().monday.at("02:00").do(weekly_retrain_job)
    print("\nScheduler running... Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
