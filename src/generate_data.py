import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataGenerator:
    def __init__(self, num_samples=5000, random_state=42):
        self.num_samples = num_samples
        np.random.seed(random_state)
        random.seed(random_state)
        self.cities = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.6139, 77.2090),
            'Bangalore': (12.9716, 77.5946),
            'Hyderabad': (17.3850, 78.4867),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462),
            'Kanpur': (26.4499, 80.3319),
            'Nagpur': (21.1458, 79.0882),
            'Indore': (22.7196, 75.8577),
            'Thane': (19.2183, 72.9781),
            'Bhopal': (23.2599, 77.4126),
            'Visakhapatnam': (17.6868, 83.2185),
            'Patna': (25.5941, 85.1376),
            'Vadodara': (22.3072, 73.1812),
            'Surat': (21.1702, 72.8311),
            'Coimbatore': (11.0168, 76.9558)
        }
        self.vehicle_types = ['Van', 'Truck', 'Motorcycle', 'Car']
        self.weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy']
        self.road_types = ['Highway', 'City', 'Rural']
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def generate_dataset(self):
        data = []
        start_date = datetime(2023, 1, 1)
        city_list = list(self.cities.keys())
        
        for i in range(self.num_samples):
            dlv_id = f"DLV{str(i+1).zfill(6)}"
            src_city = random.choice(city_list)
            dst_city = random.choice([c for c in city_list if c != src_city])
            src_lat, src_lon = self.cities[src_city]
            dst_lat, dst_lon = self.cities[dst_city]
            src_lat += np.random.uniform(-0.5, 0.5)
            src_lon += np.random.uniform(-0.5, 0.5)
            dst_lat += np.random.uniform(-0.5, 0.5)
            dst_lon += np.random.uniform(-0.5, 0.5)
            dist = self.haversine_distance(src_lat, src_lon, dst_lat, dst_lon)
            vtype = random.choice(self.vehicle_types)
            drv_id = f"DRV{random.randint(1, 100):03d}"
            wt = round(np.random.lognormal(2, 1), 2)
            wt = min(wt, 1000)
            order_tm = start_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            wthr = random.choice(self.weather_conditions)
            road = random.choice(self.road_types)
            traffic_level = random.randint(1, 10)
            speed_map = {'Highway': 80, 'City': 40, 'Rural': 60}
            base_speed = speed_map[road]
            w_factor = {'Clear': 1.0, 'Cloudy': 1.1, 'Rain': 1.3, 'Fog': 1.4, 'Snow': 1.6}
            t_factor = 1 + (traffic_level / 20)
            est_hours = (dist / base_speed) * w_factor[wthr] * t_factor
            scheduled_delivery = order_tm + timedelta(hours=est_hours)
            delay_prob = 0
            
            if traffic_level > 7:
                delay_prob += 0.3
            elif traffic_level > 4:
                delay_prob += 0.15
            
            if wthr in ['Rain', 'Snow', 'Fog']:
                delay_prob += 0.25
            
            if dist > 1000:
                delay_prob += 0.2
            elif dist > 500:
                delay_prob += 0.1
                
            if wt > 500:
                delay_prob += 0.15
                
            drv_perf = random.random()
            if drv_perf < 0.2:
                delay_prob += 0.2
                
            hr = order_tm.hour
            if 7 <= hr <= 9 or 17 <= hr <= 19:
                delay_prob += 0.15
            
            is_delayed = random.random() < min(delay_prob, 0.9)
            
            if is_delayed:
                if delay_prob > 0.6:
                    delay_mins = random.randint(60, 240)
                elif delay_prob > 0.4:
                    delay_mins = random.randint(30, 120)
                else:
                    delay_mins = random.randint(15, 60)
                actual_delivery = scheduled_delivery + timedelta(minutes=delay_mins)
            else:
                early = random.randint(-30, 10)
                actual_delivery = scheduled_delivery + timedelta(minutes=early)
            
            delay_mins = (actual_delivery - scheduled_delivery).total_seconds() / 60
            delayed_binary = 1 if delay_mins > 10 else 0
            data.append({
                'delivery_id': dlv_id,
                'source_city': src_city,
                'destination_city': dst_city,
                'source_lat': round(src_lat, 6),
                'source_long': round(src_lon, 6),
                'dest_lat': round(dst_lat, 6),
                'dest_long': round(dst_lon, 6),
                'distance_km': round(dist, 2),
                'vehicle_type': vtype,
                'driver_id': drv_id,
                'package_weight_kg': wt,
                'order_time': order_tm,
                'scheduled_delivery_time': scheduled_delivery,
                'actual_delivery_time': actual_delivery,
                'traffic_level': traffic_level,
                'weather_condition': wthr,
                'road_type': road,
                'delay_minutes': round(delay_mins, 2),
                'is_delayed': delayed_binary
            })
        return pd.DataFrame(data)
    
    def save_dataset(self, output_path):
        print(f"Generating {self.num_samples} records...")
        df = self.generate_dataset()
        df.to_csv(output_path, index=False)
        print(f"Records: {df.shape[0]}, Delay: {df['is_delayed'].mean()*100:.1f}%")
        avg_delay = df[df['is_delayed']==1]['delay_minutes'].mean()
        print(f"  Average delay: {avg_delay:.1f} minutes")
        print(f"  Saved to: {output_path}")
        return df

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("GENERATING SYNTHETIC LOGISTICS DATASET")
        print("="*70)
        output_path = "data/raw/logistics_data.csv"
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generator = DataGenerator(num_samples=5000)
        df = generator.save_dataset(output_path)
        print("\nData generation complete")
        print(f"File: {output_path}")
        print(f"Total: {len(df)} records")
        print(f"Delay rate: {df['is_delayed'].mean()*100:.1f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
