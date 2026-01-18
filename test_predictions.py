import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from datetime import datetime, timedelta
from realtime_pipeline import RealtimePredictionPipeline
import json

def test_realtime_predictions():
    print("\nTesting predictions...")
    print("-" * 40)
    try:
        pipeline = RealtimePredictionPipeline()
    except:
        print("Models not loaded yet")
        return
    print("\nTest 1: High risk (long distance, rain)")
    d1 = {
        'distance_km': 250.0, 'package_weight_kg': 45.0, 'traffic_level': 9,
        'order_time': datetime.now(), 
        'scheduled_delivery_time': datetime.now() + timedelta(hours=5),
        'vehicle_type': 'Van', 'weather_condition': 'Rain', 'road_type': 'City'
    }
    print("Info: distance={}, weight={}, traffic={}, weather={}".format(
        d1['distance_km'], d1['package_weight_kg'], d1['traffic_level'], d1['weather_condition']))
    r1 = pipeline.predict(d1)
    status = "Delayed" if r1['is_delayed'] else "On-time"
    print("Result:", status, "({:.1f}% prob)".format(r1['delay_probability']*100))
    print("\nTest 2: Low risk")
    d2 = {
        'distance_km': 50.0, 'package_weight_kg': 10.0, 'traffic_level': 3,
        'order_time': datetime.now(),
        'scheduled_delivery_time': datetime.now() + timedelta(hours=2),
        'vehicle_type': 'Motorcycle', 'weather_condition': 'Clear', 'road_type': 'Highway'
    }
    print("Info: distance={}, weight={}, traffic={}, weather={}".format(
        d2['distance_km'], d2['package_weight_kg'], d2['traffic_level'], d2['weather_condition']))
    r2 = pipeline.predict(d2)
    status = "Delayed" if r2['is_delayed'] else "On-time"
    print("Result:", status, "({:.1f}% prob)".format(r2['delay_probability']*100))
    print("\nTest 3: Medium risk")
    d3 = {
        'distance_km': 150.0, 'package_weight_kg': 25.0, 'traffic_level': 6,
        'order_time': datetime.now(),
        'scheduled_delivery_time': datetime.now() + timedelta(hours=4),
        'vehicle_type': 'Truck', 'weather_condition': 'Snow', 'road_type': 'City'
    }
    print("Info: distance={}, weight={}, traffic={}, weather={}".format(
        d3['distance_km'], d3['package_weight_kg'], d3['traffic_level'], d3['weather_condition']))
    r3 = pipeline.predict(d3)
    status = "Delayed" if r3['is_delayed'] else "On-time"
    print("Result:", status, "({:.1f}% prob)".format(r3['delay_probability']*100))
    pipeline.log_prediction(r1)
    pipeline.log_prediction(r2)
    pipeline.log_prediction(r3)
    print("\nPredictions logged to logs/prediction_log.jsonl")

if __name__ == "__main__":
    test_realtime_predictions()
