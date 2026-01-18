import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DelayPredictionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_preprocess_data(self):
        print("Loading:", self.data_path)
        df = pd.read_csv(self.data_path)
        df['order_time'] = pd.to_datetime(df['order_time'])
        df['scheduled_delivery_time'] = pd.to_datetime(df['scheduled_delivery_time'])
        df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])
        df['order_hour'] = df['order_time'].dt.hour
        df['order_day_of_week'] = df['order_time'].dt.dayofweek
        df['order_month'] = df['order_time'].dt.month
        morning_rush = (df['order_hour'] >= 7) & (df['order_hour'] <= 9)
        evening_rush = (df['order_hour'] >= 17) & (df['order_hour'] <= 19)
        df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
        df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_season'] = df['order_month'].isin([11, 12]).astype(int)
        sched_delta = (df['scheduled_delivery_time'] - df['order_time']).dt.total_seconds()
        df['scheduled_duration_hours'] = sched_delta / 3600
        df['distance_category'] = pd.cut(
            df['distance_km'], 
            bins=[0, 100, 500, 1000, 5000], 
            labels=[0, 1, 2, 3]
        )
        df['weight_category'] = pd.cut(
            df['package_weight_kg'], 
            bins=[0, 10, 50, 200, 1000], 
            labels=[0, 1, 2, 3]
        )
        for col in ['vehicle_type', 'weather_condition', 'road_type']:
            enc = LabelEncoder()
            df[f'{col}_encoded'] = enc.fit_transform(df[col])
            self.label_encoders[col] = enc
        self.feature_columns = [
            'distance_km', 'package_weight_kg', 'traffic_level',
            'order_hour', 'order_day_of_week', 'order_month',
            'scheduled_duration_hours',
            'is_rush_hour', 'is_weekend', 'is_peak_season',
            'distance_category', 'weight_category',
            'vehicle_type_encoded', 'weather_condition_encoded', 'road_type_encoded'
        ]
        X = df[self.feature_columns]
        y = df['is_delayed']
        print("Loaded {} records".format(df.shape[0]))
        print("Class dist:", y.value_counts().to_dict())
        return X, y, df
    
    def train_models(self):
        print("\nTraining models...")
        X, y, df = self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.X_test = X_test_scaled
        self.y_test = y_test
        print("\n[1] Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        print("Random Forest training done.")
        print("[2] XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1, random_state=42,
            n_jobs=-1, eval_metric='logloss', verbosity=1
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        print("XGBoost done")
        print("[3] Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=250, max_depth=7, learning_rate=0.05,
            subsample=0.8, min_samples_split=5, min_samples_leaf=2,
            random_state=42, verbose=1
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gradient_boosting'] = gb_model
        print("GB done")
        print("\nAll models ready")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_models(self):
        print("\nEvaluation Results:")
        results = {}
        for model_name, model in self.models.items():
            print("\n" + model_name + ":")
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            cm = confusion_matrix(self.y_test, y_pred)
            results[model_name] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist()
            }
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f} (important for catching delays!)")
            print(f"F1 Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"                 No    Yes")
            print(f"Actual No     {cm[0][0]:5d} {cm[0][1]:5d}")
            print(f"       Yes    {cm[1][0]:5d} {cm[1][1]:5d}")
            print(f"\nClassification Report:")
            print(classification_report(
                self.y_test, y_pred, 
                target_names=['On-Time', 'Delayed'],
                digits=4
            ))
        print("="*60)
        return results
    def plot_confusion_matrices(self, save_path='logs/confusion_matrices.png'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, save_path='logs/feature_importance.png'):
        rf_model = self.models['random_forest']
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature', hue='feature', legend=False, palette='viridis')
        plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance saved: {save_path}")
        plt.close()
    
    def save_models(self, model_dir='models/'):
        print("\nSaving models...")
        for model_name, model in self.models.items():
            path = f"{model_dir}{model_name}_model.joblib"
            joblib.dump(model, path)
            print(f"  {model_name}")
        joblib.dump(self.scaler, f"{model_dir}scaler.joblib")
        joblib.dump(self.label_encoders, f"{model_dir}label_encoders.joblib")
        with open(f"{model_dir}feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f)
        metadata = {
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_columns': self.feature_columns,
            'models': list(self.models.keys())
        }
        metadata_path = f"{model_dir}model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata")
        print("\nAll models saved successfully.")

def main():
    data_file = 'data/raw/logistics_data.csv'
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found at {data_file}")
        print("\nTo fix this:")
        print("1. Run: python src/generate_data.py")
        print("2. Then run: python src/train_models.py")
        return
    try:
        print("\n" + "="*70)
        print("STARTING MODEL TRAINING PIPELINE")
        print("="*70)
        trainer = DelayPredictionModel(data_file)
        print("\n[Step 1/5] Loading and preprocessing data...")
        trainer.train_models()
        print("\n[Step 2/5] Evaluating models...")
        results = trainer.evaluate_models()
        print("\n[Step 3/5] Plotting confusion matrices...")
        trainer.plot_confusion_matrices()
        print("\n[Step 4/5] Plotting feature importance...")
        trainer.plot_feature_importance()
        print("\n[Step 5/5] Saving models...")
        trainer.save_models()
        os.makedirs('logs', exist_ok=True)
        with open('logs/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("\n" + "="*70)
        print("[DONE] TRAINING COMPLETE!")
        print("="*70)
        print("Evaluation results saved to: logs/evaluation_results.json")
        print("\nYou can now run the dashboard:")
        print("  streamlit run app.py")
        print("="*70)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
