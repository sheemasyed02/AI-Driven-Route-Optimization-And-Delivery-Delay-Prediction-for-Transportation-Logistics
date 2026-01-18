import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from optimize_routes import RouteOptimizer, Location
from realtime_api import RealtimeDataService

class RealtimePredictionPipeline:
    def __init__(self, model_dir='models/', weather_api_key=None):
        self.model_dir = model_dir
        self.models_loaded = False
        self.models = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        try:
            self.realtime_service = RealtimeDataService(weather_api_key)
            self.load_models()
            self.models_loaded = True
        except FileNotFoundError as e:
            print(f"Warning: Models not found - {str(e)}")
            self.models_loaded = False
        
    def load_models(self):
        req_files = [
            f'{self.model_dir}random_forest_model.joblib',
            f'{self.model_dir}xgboost_model.joblib',
            f'{self.model_dir}gradient_boosting_model.joblib',
            f'{self.model_dir}scaler.joblib',
            f'{self.model_dir}label_encoders.joblib',
            f'{self.model_dir}feature_columns.json'
        ]
        missing = [f for f in req_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Missing: {missing}")
        self.models = {
            'random_forest': joblib.load(f'{self.model_dir}random_forest_model.joblib'),
            'xgboost': joblib.load(f'{self.model_dir}xgboost_model.joblib'),
            'gradient_boosting': joblib.load(f'{self.model_dir}gradient_boosting_model.joblib')
        }
        self.scaler = joblib.load(f'{self.model_dir}scaler.joblib')
        self.label_encoders = joblib.load(f'{self.model_dir}label_encoders.joblib')
        with open(f'{self.model_dir}feature_columns.json', 'r') as f:
            self.feature_columns = json.load(f)

    def get_realtime_conditions(self, lat, lon):
        # get current weather and traffic data
        return self.realtime_service.get_conditions(lat, lon)
    
    def preprocess_input(self, delivery_data):
        if isinstance(delivery_data['order_time'], str):
            order_time = pd.to_datetime(delivery_data['order_time'])
        else:
            order_time = delivery_data['order_time']
        if isinstance(delivery_data['scheduled_delivery_time'], str):
            scheduled_time = pd.to_datetime(delivery_data['scheduled_delivery_time'])
        else:
            scheduled_time = delivery_data['scheduled_delivery_time']
        order_hour = order_time.hour
        order_day_of_week = order_time.weekday()
        order_month = order_time.month
        is_rush_hour = 1 if (7 <= order_hour <= 9 or 17 <= order_hour <= 19) else 0
        is_weekend = 1 if order_day_of_week in [5, 6] else 0
        is_peak_season = 1 if order_month in [11, 12] else 0
        scheduled_duration_hours = (scheduled_time - order_time).total_seconds() / 3600
        distance_km = delivery_data['distance_km']
        if distance_km <= 100:
            distance_category = 0
        elif distance_km <= 500:
            distance_category = 1
        elif distance_km <= 1000:
            distance_category = 2
        else:
            distance_category = 3
        weight_kg = delivery_data['package_weight_kg']
        if weight_kg <= 10:
            weight_category = 0
        elif weight_kg <= 50:
            weight_category = 1
        elif weight_kg <= 200:
            weight_category = 2
        else:
            weight_category = 3
        v_enc = self.label_encoders['vehicle_type'].transform(
            [delivery_data['vehicle_type']]
        )[0]
        w_enc = self.label_encoders['weather_condition'].transform(
            [delivery_data['weather_condition']]
        )[0]
        r_enc = self.label_encoders['road_type'].transform(
            [delivery_data['road_type']]
        )[0]
        features = np.array([[
            delivery_data['distance_km'],
            delivery_data['package_weight_kg'],
            delivery_data['traffic_level'],
            order_hour,
            order_day_of_week,
            order_month,
            scheduled_duration_hours,
            is_rush_hour,
            is_weekend,
            is_peak_season,
            distance_category,
            weight_category,
            v_enc,
            w_enc,
            r_enc
        ]])
        features_scaled = self.scaler.transform(features)
        return features_scaled
    
    def predict_delay(self, delivery_data, model_name='xgboost'):
        features = self.preprocess_input(delivery_data)
        model = self.models[model_name]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        result = {
            'is_delayed': bool(prediction),
            'delay_probability': float(probability[1]),
            'on_time_probability': float(probability[0]),
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': model_name,
            'delivery_data': delivery_data
        }
        return result
    
    def predict(self, delivery_data):
        features = self.preprocess_input(delivery_data)
        predictions = []
        probabilities = []
        for model in self.models.values():
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]
            predictions.append(pred)
            probabilities.append(prob)
        avg_prob = np.mean(probabilities)
        final_pred = 1 if avg_prob > 0.5 else 0
        result = {
            'is_delayed': bool(final_pred),
            'delay_probability': float(avg_prob),
            'on_time_probability': float(1 - avg_prob),
            'individual_predictions': {
                'random_forest': {'delayed': bool(predictions[0]), 'prob': float(probabilities[0])},
                'xgboost': {'delayed': bool(predictions[1]), 'prob': float(probabilities[1])},
                'gradient_boosting': {'delayed': bool(predictions[2]), 'prob': float(probabilities[2])}
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'delivery_data': delivery_data
        }
        return result
    
    def optimize_route_for_delivery(self, locations, algorithm='ortools'):
        location_objects = [
            Location(
                id=idx,
                name=loc['name'],
                lat=loc['lat'],
                lon=loc['lon'],
                time_window=loc.get('time_window', (0, 1440))
            )
            for idx, loc in enumerate(locations)
        ]
        optimizer = RouteOptimizer()
        result = optimizer.optimize_route(
            location_objects,
            algorithm=algorithm,
            depot_idx=0,
            num_vehicles=1
        )
        return result
    
    def process_delivery(self, delivery_data, locations=None):
        print("\nProcessing delivery...")
        delay_pred = self.predict(delivery_data)
        status = "Delayed" if delay_pred['is_delayed'] else "On-time"
        print(f"Status: {status}")
        print(f"Delay probability: {delay_pred['delay_probability']*100:.1f}%")
        route_opt = None
        if locations:
            print("Optimizing route...")
            route_opt = self.optimize_route_for_delivery(locations)
            if route_opt:
                dist = route_opt['total_distance_km']
                print(f"Total distance: {dist:.2f} km")
        self.log_prediction(delay_pred, route_opt)
        
        return {
            'delay_prediction': delay_pred,
            'route_optimization': route_opt,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def log_prediction(self, delay_prediction, route_optimization=None):
        pred_copy = delay_prediction.copy()
        if 'delivery_data' in pred_copy:
            dd = pred_copy['delivery_data'].copy()
            for k, v in dd.items():
                if hasattr(v, 'strftime'):
                    dd[k] = v.strftime('%Y-%m-%d %H:%M:%S')
            pred_copy['delivery_data'] = dd
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'delay_prediction': pred_copy,
            'route_optimization': route_optimization
        }
        log_file = 'logs/prediction_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

def demo_realtime_pipeline():
    delivery_data = {
        'distance_km': 150.5,
        'package_weight_kg': 25.0,
        'traffic_level': 8,
        'order_time': datetime(2024, 3, 15, 10, 30),
        'scheduled_delivery_time': datetime(2024, 3, 15, 14, 30),
        'vehicle_type': 'Van',
        'weather_condition': 'Rain',
        'road_type': 'City'
    }
    
    locations = [
        {'name': 'Warehouse', 'lat': 40.7128, 'lon': -74.0060},
        {'name': 'Customer A', 'lat': 40.7580, 'lon': -73.9855},
        {'name': 'Customer B', 'lat': 40.7489, 'lon': -73.9680},
        {'name': 'Customer C', 'lat': 40.7614, 'lon': -73.9776}
    ]
    pipeline = RealtimePredictionPipeline()
    result = pipeline.process_delivery(delivery_data, locations)
    print("\nProcessing complete.")
    return result

if __name__ == "__main__":
    demo_realtime_pipeline()
