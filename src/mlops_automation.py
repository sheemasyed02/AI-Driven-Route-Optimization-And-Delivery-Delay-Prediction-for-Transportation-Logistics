import os
import shutil
import json
import joblib
import pandas as pd
from datetime import datetime
from train_models import DelayPredictionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

class MLOpsManager:
    def __init__(self, data_path='data/raw/logistics_data.csv', 
                 model_dir='models/', 
                 version_dir='models/versions/',
                 log_dir='logs/'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.version_dir = version_dir
        self.log_dir = log_dir
        os.makedirs(self.version_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def get_current_model_performance(self):
        try:
            models = {
                'random_forest': joblib.load(f'{self.model_dir}random_forest_model.joblib'),
                'xgboost': joblib.load(f'{self.model_dir}xgboost_model.joblib'),
                'gradient_boosting': joblib.load(f'{self.model_dir}gradient_boosting_model.joblib')
            }
            scaler = joblib.load(f'{self.model_dir}scaler.joblib')
            df = pd.read_csv(self.data_path)
            df['order_time'] = pd.to_datetime(df['order_time'])
            df['scheduled_delivery_time'] = pd.to_datetime(df['scheduled_delivery_time'])
            df['order_hour'] = df['order_time'].dt.hour
            df['order_day_of_week'] = df['order_time'].dt.dayofweek
            df['order_month'] = df['order_time'].dt.month
            df['scheduled_duration_hours'] = (
                df['scheduled_delivery_time'] - df['order_time']
            ).dt.total_seconds() / 3600
            with open(f'{self.model_dir}feature_columns.json', 'r') as f:
                feature_columns = json.load(f)
            label_encoders = joblib.load(f'{self.model_dir}label_encoders.joblib')
            for col in ['vehicle_type', 'weather_condition', 'road_type']:
                df[f'{col}_encoded'] = label_encoders[col].transform(df[col])
            test_size = int(len(df) * 0.2)
            df_test = df.tail(test_size)
            X_test = df_test[feature_columns]
            y_test = df_test['is_delayed']
            X_test_scaled = scaler.transform(X_test)
            performance = {}
            for model_name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                performance[model_name] = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred)),
                    'recall': float(recall_score(y_test, y_pred)),
                    'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
                }
            return performance
        
        except FileNotFoundError as e:
            print(f"⚠ Models not found: {e}")
            print("Need to train initial models first")
            return None
    
    def retrain_models(self):
        print(f"\nRetraining models at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model_trainer = DelayPredictionModel(self.data_path)
        model_trainer.train_models()
        new_performance = model_trainer.evaluate_models()
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_model_version(version)
        model_trainer.save_models(self.model_dir)
        model_trainer.plot_confusion_matrices(f'{self.log_dir}confusion_matrices_{version}.png')
        model_trainer.plot_feature_importance(f'{self.log_dir}feature_importance_{version}.png')
        print(f"Retraining complete, version: {version}")
        return new_performance, version
    
    def save_model_version(self, version):
        version_path = os.path.join(self.version_dir, version)
        os.makedirs(version_path, exist_ok=True)
        for filename in os.listdir(self.model_dir):
            if filename.endswith(('.joblib', '.json')):
                src = os.path.join(self.model_dir, filename)
                dst = os.path.join(version_path, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        print(f"Version saved: {version_path}")
    
    def compare_performance(self, old_perf, new_perf):
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON (Old vs New)")
        print("="*60)
        improvements = {}
        for model_name in old_perf.keys():
            old = old_perf[model_name]
            new = new_perf[model_name]
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            auc_change = new['roc_auc'] - old['roc_auc']
            acc_change = new['accuracy'] - old['accuracy']
            rec_change = new['recall'] - old['recall']
            print(f"  ROC-AUC:  {old['roc_auc']:.4f} → {new['roc_auc']:.4f} ({auc_change:+.4f})")
            print(f"  Accuracy: {old['accuracy']:.4f} → {new['accuracy']:.4f} ({acc_change:+.4f})")
            print(f"  Recall:   {old['recall']:.4f} → {new['recall']:.4f} ({rec_change:+.4f})")
            improved = new['roc_auc'] > old['roc_auc']
            status = "✓ IMPROVED" if improved else "✗ DEGRADED"
            print(f"  Status: {status}")
            improvements[model_name] = improved
        print("\n" + "="*60)
        return improvements
    
    def log_training_session(self, old_performance, new_performance, version):
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': version,
            'old_performance': old_performance,
            'new_performance': new_performance,
            'data_path': self.data_path
        }
        log_file = os.path.join(self.log_dir, 'training_log.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        print(f"Training session logged: {log_file}")
    
    def check_and_retrain(self, performance_threshold=0.85):
        print("\n" + "="*60)
        print("MLOps AUTOMATED PERFORMANCE CHECK")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Threshold: {performance_threshold}")
        old_perf = self.get_current_model_performance()
        if old_perf is None:
            print("\n[WARNING] No existing models found")
            print("Starting initial model training...")
            new_perf, version = self.retrain_models()
            self.log_training_session({}, new_perf, version)
            return
        
        needs_retrain = False
        print("\nModel Performance Status:")
        print("-" * 60)
        for model_name, metrics in old_perf.items():
            roc_auc = metrics['roc_auc']
            status = "✓ PASS" if roc_auc >= performance_threshold else "✗ FAIL"
            print(f"  {model_name:20s} ROC-AUC: {roc_auc:.4f}  [{status}]")
            if roc_auc < performance_threshold:
                needs_retrain = True
        print("-" * 60)
        if needs_retrain:
            print(f"\n⚠ One or more models below threshold ({performance_threshold})")
            print("Starting retraining process...")
            new_perf, version = self.retrain_models()
            self.compare_performance(old_perf, new_perf)
            self.log_training_session(old_perf, new_perf, version)
        else:
            print(f"\n✓ All models meet threshold. No retraining needed.")
        print("\n" + "="*60)
    
    def generate_performance_report(self):
        try:
            log_file = os.path.join(self.log_dir, 'training_log.jsonl')
            if not os.path.exists(log_file):
                print("No training history available.")
                return
            entries = []
            with open(log_file, 'r') as f:
                for line in f:
                    entries.append(json.loads(line))
            print("\nPerformance History Report")
            for idx, entry in enumerate(entries, 1):
                print(f"\nSession {idx} - {entry['timestamp']}")
                print(f"Version: {entry['version']}")
                if entry['new_performance']:
                    for model_name, metrics in entry['new_performance'].items():
                        print(f"{model_name}: ROC-AUC = {metrics['roc_auc']:.4f}")
            report_file = os.path.join(self.log_dir, 'performance_report.json')
            with open(report_file, 'w') as f:
                json.dump(entries, f, indent=4)
            print(f"\nReport saved to: {report_file}")
        except Exception as e:
            print(f"Error generating report: {e}")

def main():
    mlops = MLOpsManager()
    mlops.check_and_retrain(performance_threshold=0.85)
    mlops.generate_performance_report()

if __name__ == "__main__":
    main()

