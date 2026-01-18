# AI-Driven Route Optimization and Delivery Delay Prediction

Project for predicting delivery delays and optimizing routes using machine learning.

## What this does

Uses 3 machine learning models to predict if a delivery will be delayed. Also optimizes routes using different algorithms. Has a dashboard to view predictions and analytics.

## How to run

Requirements: Python 3.8 or higher

Install dependencies:
```
pip install -r requirements.txt
```

Generate training data:
```
python src/generate_data.py
```

Train the models:
```
python src/train_models.py
```

Start the dashboard:
```
streamlit run app.py
```

The dashboard will open at http://localhost:8501

## Project layout

data/ - contains the logistics dataset
models/ - trained models stored here
logs/ - model evaluation results and prediction logs
src/ - source code for data generation, training, prediction

## Models used

Random Forest Classifier
XGBoost Classifier
Gradient Boosting Classifier

## Routing

Three routing algorithms available:
- Dijkstra's algorithm
- A* search
- Google OR-Tools VRP solver

## Dashboard pages

Overview - shows delivery statistics and metrics
Predictions - make predictions on new deliveries
Routes - optimize delivery routes
Analytics - driver performance and trends
Models - view model performance metrics

## Automatic retraining

Models will automatically retrain if performance drops below 0.85 ROC-AUC. This runs weekly on Mondays at 2 AM.

To manually trigger retraining:
```
python src/mlops_automation.py
```
