import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import json
import hashlib
from datetime import datetime, timedelta
import joblib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from realtime_pipeline import RealtimePredictionPipeline
from optimize_routes import RouteOptimizer, Location
from realtime_api import RealtimeDataService
st.set_page_config(page_title="Logistics Dashboard", layout="wide")

INDIAN_CITIES = {
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Guntur": (16.3067, 80.4365),
    "Tirupati": (13.6288, 79.4192),
    "Nellore": (14.4426, 79.9865),
    "Kurnool": (15.8281, 78.0373),
    "Rajahmundry": (17.0005, 81.8040),
    "Kakinada": (16.9891, 82.2475),
    "Eluru": (16.7107, 81.0952),
    "Anantapur": (14.6819, 77.6006),
    "Guwahati": (26.1445, 91.7362),
    "Silchar": (24.8333, 92.7789),
    "Dibrugarh": (27.4728, 94.9111),
    "Jorhat": (26.7509, 94.2037),
    "Tezpur": (26.6338, 92.7926),
    "Patna": (25.5941, 85.1376),
    "Gaya": (24.7956, 85.0077),
    "Muzaffarpur": (26.1197, 85.3910),
    "Bhagalpur": (25.2425, 86.9842),
    "Darbhanga": (26.1542, 85.8918),
    "Purnia": (25.7771, 87.4753),
    "Raipur": (21.2514, 81.6296),
    "Bhilai": (21.1938, 81.3509),
    "Bilaspur": (22.0797, 82.1391),
    "Durg": (21.1904, 81.2849),
    "Korba": (22.3595, 82.7501),
    "New Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Ghaziabad": (28.6692, 77.4538),
    "Gurugram": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178),
    "Panaji": (15.4909, 73.8278),
    "Margao": (15.2832, 73.9862),
    "Vasco da Gama": (15.3982, 73.8113),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Rajkot": (22.3039, 70.8022),
    "Gandhinagar": (23.2156, 72.6369),
    "Bhavnagar": (21.7645, 72.1519),
    "Jamnagar": (22.4707, 70.0577),
    "Junagadh": (21.5222, 70.4579),
    "Anand": (22.5645, 72.9289),
    "Bharuch": (21.7051, 72.9959),
    "Morbi": (22.8173, 70.8370),
    "Ambala": (30.3782, 76.7767),
    "Hisar": (29.1492, 75.7217),
    "Rohtak": (28.8955, 76.6066),
    "Panipat": (29.3909, 76.9635),
    "Karnal": (29.6857, 76.9905),
    "Sonipat": (28.9931, 77.0151),
    "Yamunanagar": (30.1290, 77.2674),
    "Shimla": (31.1048, 77.1734),
    "Manali": (32.2396, 77.1887),
    "Dharamshala": (32.2190, 76.3234),
    "Solan": (30.9045, 77.0967),
    "Mandi": (31.7080, 76.9318),
    "Srinagar": (34.0837, 74.7973),
    "Jammu": (32.7266, 74.8570),
    "Anantnag": (33.7311, 75.1487),
    "Baramulla": (34.2090, 74.3614),
    "Kathua": (32.3896, 75.5230),
    "Ranchi": (23.3441, 85.3096),
    "Jamshedpur": (22.8046, 86.2029),
    "Dhanbad": (23.7957, 86.4304),
    "Bokaro": (23.6693, 85.9845),
    "Hazaribagh": (23.9925, 85.3617),
    "Bangalore": (12.9716, 77.5946),
    "Mysore": (12.2958, 76.6394),
    "Hubli": (15.3647, 75.1240),
    "Mangalore": (12.9141, 74.8560),
    "Belgaum": (15.8497, 74.4977),
    "Davangere": (14.4644, 75.9218),
    "Bellary": (15.1394, 76.9214),
    "Gulbarga": (17.3297, 76.8343),
    "Tumkur": (13.3409, 77.1011),
    "Shimoga": (13.9299, 75.5681),
    "Bijapur": (16.8302, 75.7100),
    "Udupi": (13.3409, 74.7421),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
    "Thrissur": (10.5276, 76.2144),
    "Kollam": (8.8932, 76.6141),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0510, 76.0711),
    "Kannur": (11.8745, 75.3704),
    "Kasaragod": (12.4996, 74.9869),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Jabalpur": (23.1815, 79.9864),
    "Gwalior": (26.2183, 78.1828),
    "Ujjain": (23.1765, 75.7885),
    "Sagar": (23.8388, 78.7378),
    "Satna": (24.5854, 80.8322),
    "Rewa": (24.5373, 81.3042),
    "Dewas": (22.9623, 76.0508),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Solapur": (17.6599, 75.9064),
    "Kolhapur": (16.7050, 74.2433),
    "Thane": (19.2183, 72.9781),
    "Amravati": (20.9374, 77.7796),
    "Akola": (20.7002, 77.0082),
    "Jalgaon": (21.0077, 75.5626),
    "Nanded": (19.1383, 77.3210),
    "Latur": (18.4088, 76.5604),
    "Dhule": (20.9042, 74.7749),
    "Ahmednagar": (19.0952, 74.7496),
    "Imphal": (24.8170, 93.9368),
    "Shillong": (25.5788, 91.8933),
    "Aizawl": (23.7271, 92.7176),
    "Kohima": (25.6751, 94.1086),
    "Dimapur": (25.9044, 93.7265),
    "Bhubaneswar": (20.2961, 85.8245),
    "Cuttack": (20.4625, 85.8830),
    "Rourkela": (22.2604, 84.8536),
    "Sambalpur": (21.4669, 83.9812),
    "Brahmapur": (19.3149, 84.7941),
    "Puri": (19.8135, 85.8312),
    "Chandigarh": (30.7333, 76.7794),
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
    "Jalandhar": (31.3260, 75.5762),
    "Patiala": (30.3398, 76.3869),
    "Bathinda": (30.2110, 74.9455),
    "Mohali": (30.7046, 76.7179),
    "Jaipur": (26.9124, 75.7873),
    "Jodhpur": (26.2389, 73.0243),
    "Udaipur": (24.5854, 73.7125),
    "Kota": (25.2138, 75.8648),
    "Ajmer": (26.4499, 74.6399),
    "Bikaner": (28.0229, 73.3119),
    "Alwar": (27.5530, 76.6346),
    "Bharatpur": (27.2152, 77.5030),
    "Sikar": (27.6094, 75.1399),
    "Sri Ganganagar": (29.9038, 73.8772),
    "Gangtok": (27.3389, 88.6065),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Salem": (11.6643, 78.1460),
    "Trichy": (10.7905, 78.7047),
    "Tirunelveli": (8.7139, 77.7567),
    "Vellore": (12.9165, 79.1325),
    "Erode": (11.3410, 77.7172),
    "Thoothukudi": (8.7642, 78.1348),
    "Tiruppur": (11.1085, 77.3411),
    "Thanjavur": (10.7902, 79.1378),
    "Dindigul": (10.3624, 77.9695),
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9689, 79.5941),
    "Nizamabad": (18.6725, 78.0941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Mahbubnagar": (16.7488, 77.9821),
    "Agartala": (23.8315, 91.2868),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Allahabad": (25.4358, 81.8463),
    "Meerut": (28.9845, 77.7064),
    "Bareilly": (28.3670, 79.4304),
    "Aligarh": (27.8974, 78.0880),
    "Moradabad": (28.8386, 78.7733),
    "Gorakhpur": (26.7606, 83.3732),
    "Saharanpur": (29.9680, 77.5510),
    "Mathura": (27.4924, 77.6737),
    "Jhansi": (25.4484, 78.5685),
    "Firozabad": (27.1591, 78.3957),
    "Dehradun": (30.3165, 78.0322),
    "Haridwar": (29.9457, 78.1642),
    "Rishikesh": (30.0869, 78.2676),
    "Nainital": (29.3919, 79.4542),
    "Haldwani": (29.2183, 79.5130),
    "Roorkee": (29.8543, 77.8880),
    "Kolkata": (22.5726, 88.3639),
    "Howrah": (22.5958, 88.2636),
    "Durgapur": (23.5204, 87.3119),
    "Asansol": (23.6739, 86.9524),
    "Siliguri": (26.7271, 88.3953),
    "Haldia": (22.0667, 88.0698),
    "Kharagpur": (22.3460, 87.2320),
}

def load_pipeline():
    try:
        pipeline = RealtimePredictionPipeline()
        return pipeline
    except Exception as e:
        return None

def load_data():
    try:
        df = pd.read_csv('data/raw/logistics_data.csv')
        df['order_time'] = pd.to_datetime(df['order_time'])
        df['scheduled_delivery_time'] = pd.to_datetime(df['scheduled_delivery_time'])
        df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])
        return df
    except:
        return None

def draw_map(locs, route=None, traffic_levels=None):
    if not locs:
        m = folium.Map(location=[22.0, 78.0], zoom_start=5)
        return m
    center_lat = sum(l['lat'] for l in locs) / len(locs)
    center_lon = sum(l['lon'] for l in locs) / len(locs)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    for i, loc in enumerate(locs):
        if traffic_levels and i in traffic_levels:
            lvl = traffic_levels[i]
            if lvl <= 3:
                marker_color = 'green'
            elif lvl <= 6:
                marker_color = 'orange'
            else:
                marker_color = 'red'
        else:
            marker_color = 'red' if i == 0 else 'blue'

        tooltip_text = loc['name']
        if traffic_levels and i in traffic_levels:
            tooltip_text = loc['name'] + ' | Traffic: ' + str(traffic_levels[i]) + '/10'

        popup_text = 'Stop ' + str(i) + ': ' + loc['name']
        if i == 0:
            popup_text += ' (Depot)'

        folium.Marker(
            [loc['lat'], loc['lon']],
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=tooltip_text,
            icon=folium.Icon(color=marker_color)
        ).add_to(m)
    if len(locs) > 1:
        preview_coords = [[loc['lat'], loc['lon']] for loc in locs]
        folium.PolyLine(
            preview_coords,
            color='gray',
            weight=2,
            opacity=0.5,
            dash_array='10',
            tooltip='Current stop order (preview)'
        ).add_to(m)

    if route and len(route) > 1:
        route_coords = [[locs[idx]['lat'], locs[idx]['lon']] for idx in route]
        folium.PolyLine(
            route_coords,
            color='blue',
            weight=4,
            opacity=0.8,
            tooltip='Optimized route'
        ).add_to(m)

    if len(locs) >= 2:
        lats = [l['lat'] for l in locs]
        lons = [l['lon'] for l in locs]
        m.fit_bounds([[min(lats) - 0.5, min(lons) - 0.5], [max(lats) + 0.5, max(lons) + 0.5]])
    elif len(locs) == 1:
        m.location = [locs[0]['lat'], locs[0]['lon']]

    return m

def show_dashboard(df):
    st.header("Overview")
    total = len(df)
    delayed = df['is_delayed'].sum()
    delay_pct = (delayed / total) * 100
    avg_dist = df['distance_km'].mean()
    avg_delay_mins = df[df['is_delayed']==1]['delay_minutes'].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Deliveries", f"{total:,}")
    c2.metric("Delay Rate", f"{delay_pct:.1f}%")
    c3.metric("Avg Distance", f"{avg_dist:.1f} km")
    c4.metric("Avg Delay", f"{avg_delay_mins:.0f} min")
    st.write("---")
    left, right = st.columns(2)
    with left:
        st.write("**Delays by Weather**")
        weather_data = df.groupby('weather_condition')['is_delayed'].mean() * 100
        fig = px.bar(x=weather_data.index, y=weather_data.values,
                    labels={'x': 'Weather', 'y': 'Delay %'})
        fig.update_traces(marker_color='indianred')
        st.plotly_chart(fig, width='stretch')
    with right:
        st.write("**Delays by Traffic**")
        traffic_data = df.groupby('traffic_level')['is_delayed'].mean() * 100
        fig = px.line(x=traffic_data.index, y=traffic_data.values,
                     labels={'x': 'Traffic Level', 'y': 'Delay %'}, markers=True)
        st.plotly_chart(fig, width='stretch')
    left, right = st.columns(2)
    with left:
        st.write("**Vehicle Distribution**")
        veh_counts = df['vehicle_type'].value_counts()
        fig = px.pie(values=veh_counts.values, names=veh_counts.index, hole=0.3)
        st.plotly_chart(fig, width='stretch')
    with right:
        st.write("**Distance vs Delay**")
        sample = df.sample(min(500, len(df)))
        fig = px.scatter(sample, x='distance_km', y='delay_minutes', color='is_delayed',
                        color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, width='stretch')
def show_predictions(pipeline):
    st.header("Predict Delivery Delay")
    if pipeline is None or not pipeline.models_loaded:
        st.error("Models not loaded")
        st.info("""Run these commands first:   
python src/generate_data.py
python src/train_models.py

Then refresh this page.""")
        return
    
    st.write("**Check Current Conditions**")
    api = RealtimeDataService()

    city = st.selectbox("City", sorted(INDIAN_CITIES.keys()))
    lat, lon = INDIAN_CITIES[city]
    cond = api.get_conditions(lat, lon)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Weather", cond['weather']['condition'])
    mc2.metric("Temp", f"{cond['weather']['temperature']}°C")
    mc3.metric("Traffic", f"{cond['traffic']['level']}/10")
    st.write("---")
    st.write("**Enter Delivery Details**")
    c1, c2, c3 = st.columns(3)
    with c1:
        dist = st.number_input("Distance (km)", 1.0, 3000.0, 150.0)
        weight = st.number_input("Weight (kg)", 0.1, 1000.0, 25.0)
        traffic = st.slider("Traffic", 1, 10, 5)
    with c2:
        vehicle = st.selectbox("Vehicle", ["Van", "Truck", "Motorcycle", "Car"])
        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog", "Cloudy"])
        road = st.selectbox("Road Type", ["Highway", "City", "Rural"])
    with c3:
        date = st.date_input("Date", datetime.now())
        time = st.time_input("Time", datetime.now().time())
        hours = st.number_input("Delivery Window (hrs)", 1, 24, 4)
    if st.button("Check Delay Risk"):
        order_dt = datetime.combine(date, time)
        sched_dt = order_dt + timedelta(hours=hours)
        data = {
            'distance_km': dist,
            'package_weight_kg': weight,
            'traffic_level': traffic,
            'order_time': order_dt,
            'scheduled_delivery_time': sched_dt,
            'vehicle_type': vehicle,
            'weather_condition': weather,
            'road_type': road
        }
        res = pipeline.predict(data)
        st.write("---")
        prob = res['delay_probability'] * 100
        rc1, rc2 = st.columns(2)
        with rc1:
            if res['is_delayed']:
                st.error(f"High delay risk: {prob:.1f}%")
            else:
                st.success(f"Low delay risk: {prob:.1f}%")
        with rc2:
            st.write("Model breakdown:")
            for name, pred in res['individual_predictions'].items():
                status = "Delayed" if pred['delayed'] else "On-time"
                st.write(f"- {name}: {status} ({pred['prob']*100:.0f}%)")

def show_routes():
    st.header("Optimize Delivery Route")
    if 'route_locs' not in st.session_state:
        st.session_state.route_locs = []
    if 'route_result' not in st.session_state:
        st.session_state.route_result = None
    if 'route_gen' not in st.session_state:
        st.session_state.route_gen = 0
    editor_col, map_col = st.columns([1, 1])
    with editor_col:
        st.write("**Current stops** (stop 0 is the depot)")
        gen = str(st.session_state.route_gen)
        city_names = sorted(INDIAN_CITIES.keys())

        if len(st.session_state.route_locs) == 0:
            st.info("No stops added yet. Use the button below to add stops.")

        to_remove = None
        for i, loc in enumerate(st.session_state.route_locs):
            row = st.columns([4, 2, 2, 1])
            current_name = loc['name']
            if current_name in city_names:
                default_idx = city_names.index(current_name)
            else:
                default_idx = 0
            chosen = row[0].selectbox(
                "Stop " + str(i) + (" (Depot)" if i == 0 else ""),
                city_names,
                index=default_idx,
                key="rn_" + gen + "_" + str(i)
            )
            new_lat, new_lon = INDIAN_CITIES[chosen]
            if chosen != loc['name']:
                loc['name'] = chosen
                loc['lat'] = new_lat
                loc['lon'] = new_lon
            row[1].text_input("Lat", value=str(round(loc['lat'], 4)), key="rl_" + gen + "_" + str(i), disabled=True)
            row[2].text_input("Lon", value=str(round(loc['lon'], 4)), key="ro_" + gen + "_" + str(i), disabled=True)
            if len(st.session_state.route_locs) > 1:
                if row[3].button("Remove", key="rmv_" + gen + "_" + str(i)):
                    to_remove = i

        if to_remove is not None:
            st.session_state.route_locs.pop(to_remove)
            st.session_state.route_result = None
            st.session_state.route_gen += 1
            st.rerun()

        add_col, clear_col = st.columns(2)
        with add_col:
            if st.button("Add stop"):
                first_city = city_names[0]
                lat, lon = INDIAN_CITIES[first_city]
                st.session_state.route_locs.append({'name': first_city, 'lat': lat, 'lon': lon})
                st.session_state.route_result = None
                st.session_state.route_gen += 1
                st.rerun()
        with clear_col:
            if len(st.session_state.route_locs) > 0 and st.button("Clear all stops"):
                st.session_state.route_locs = []
                st.session_state.route_result = None
                st.session_state.route_gen += 1
                st.rerun()

    with map_col:
        st.write("**Live route map** (updates as you edit stops)")
        if st.session_state.route_result is not None:
            old_snapshot = st.session_state.route_result.get('loc_snapshot')
            if old_snapshot is not None:
                current_coords = [(loc['lat'], loc['lon']) for loc in st.session_state.route_locs]
                if current_coords != old_snapshot:
                    st.session_state.route_result = None
        saved_route = None
        saved_traffic = None
        if st.session_state.route_result is not None:
            saved_route   = st.session_state.route_result.get('route')
            saved_traffic = st.session_state.route_result.get('traffic_levels')
        live_map = draw_map(st.session_state.route_locs, saved_route, saved_traffic)
        map_state = str({
            'coords': [(l['lat'], l['lon']) for l in st.session_state.route_locs],
            'route': saved_route,
            'traffic': saved_traffic,
            'gen': st.session_state.route_gen
        })
        map_key = "map_" + hashlib.md5(map_state.encode()).hexdigest()[:8]
        st_folium(live_map, width='stretch', height=480, returned_objects=[], key=map_key)
    st.write("---")
    st.write("**Route settings**")
    settings_col, traffic_col = st.columns(2)
    with settings_col:
        algo = st.selectbox(
            "Routing algorithm",
            ["OR-Tools (recommended for multiple stops)", "Dijkstra", "A*"]
        )
    with traffic_col:
        use_realtime = st.checkbox(
            "Apply real-time traffic weights",
            value=True,
            help="Fetches live traffic level per stop and increases edge cost for congested areas."
        )
    if st.button("Find Best Route", type="primary"):
        if len(st.session_state.route_locs) < 2:
            st.warning("Add at least 2 stops before running the optimizer.")
            return
        loc_objects = []
        for i, loc in enumerate(st.session_state.route_locs):
            loc_objects.append(Location(i, loc['name'], loc['lat'], loc['lon']))
        opt = RouteOptimizer()
        opt.create_distance_matrix(loc_objects)
        fetched_traffic = {}
        if use_realtime:
            api = RealtimeDataService()
            with st.spinner("Fetching traffic conditions for each stop..."):
                for i, loc in enumerate(st.session_state.route_locs):
                    try:
                        conditions = api.get_conditions(loc['lat'], loc['lon'])
                        fetched_traffic[i] = int(conditions['traffic']['level'])
                    except Exception:
                        fetched_traffic[i] = 5
            opt.apply_traffic_weights(fetched_traffic)
        try:
            if "OR-Tools" in algo:
                result = opt.ortools_vrp(depot_idx=0, num_vehicles=1)
                if result is None:
                    st.error("OR-Tools could not find a valid route. Try Dijkstra or A*.")
                    return
                computed_route = result['routes'][0]['route']
                total_dist     = result['total_distance_km']
            elif "Dijkstra" in algo:
                total_dist, computed_route = opt.dijkstra(0, len(loc_objects) - 1)
            else:
                total_dist, computed_route = opt.astar(0, len(loc_objects) - 1)
        except Exception as err:
            st.error("Route computation failed: " + str(err))
            return
        st.session_state.route_result = {
            'route':          computed_route,
            'total_dist':     total_dist,
            'traffic_levels': fetched_traffic if use_realtime else None,
            'used_traffic':   use_realtime,
            'algo':           algo,
            'loc_snapshot':   [(loc['lat'], loc['lon']) for loc in st.session_state.route_locs],
        }
        st.rerun()
    if st.session_state.route_result is not None:
        res = st.session_state.route_result
        st.write("---")
        dist_label = "Effective distance (traffic-weighted)" if res['used_traffic'] else "Total distance"
        st.metric(dist_label, str(round(res['total_dist'], 1)) + " km")
        stop_names = [st.session_state.route_locs[i]['name'] for i in res['route']]
        st.write("Route order: " + " -> ".join(stop_names))
        if res['used_traffic'] and res['traffic_levels']:
            st.write("**Traffic conditions at each stop**")
            traffic_rows = []
            for i, loc in enumerate(st.session_state.route_locs):
                lvl = res['traffic_levels'].get(i, 5)
                if lvl <= 3:
                    condition = "Low"
                elif lvl <= 6:
                    condition = "Moderate"
                else:
                    condition = "High"
                traffic_rows.append({
                    'Stop':          loc['name'],
                    'Traffic Level': str(lvl) + "/10",
                    'Condition':     condition,
                })
            st.dataframe(pd.DataFrame(traffic_rows), hide_index=True)
def show_analytics(df):
    st.header("Analytics")
    st.write("**Top Drivers by Performance**")
    driver_perf = df.groupby('driver_id').agg({
        'is_delayed': ['count', 'mean'],
        'distance_km': 'sum'
    }).round(2)
    driver_perf.columns = ['Deliveries', 'Delay Rate', 'Total KM']
    driver_perf['Delay Rate'] = (driver_perf['Delay Rate'] * 100).round(1)
    driver_perf = driver_perf.sort_values('Delay Rate').head(15)
    st.dataframe(driver_perf)
    st.write("---")
    st.write("**Daily Trends**")
    df['date'] = df['order_time'].dt.date
    daily = df.groupby('date').agg({'is_delayed': ['count', 'sum']}).reset_index()
    daily.columns = ['Date', 'Total', 'Delayed']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Total'], name='Total', mode='lines'))
    fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Delayed'], name='Delayed',
                            mode='lines', line=dict(color='red')))
    st.plotly_chart(fig, width='stretch')
    st.write("**Traffic-Weather Heatmap**")
    heat = df.groupby(['traffic_level', 'weather_condition'])['is_delayed'].mean() * 100
    heat = heat.unstack(fill_value=0)
    fig = px.imshow(heat, labels={'color': 'Delay %'}, color_continuous_scale='OrRd')
    st.plotly_chart(fig, width='stretch')
def show_model_perf():
    st.header("Model Performance")
    try:
        with open('logs/evaluation_results.json') as f:
            results = json.load(f)
    except:
        st.error("No results found. Train models first.")
        return
    st.write("**Comparison**")
    rows = []
    for name, m in results.items():
        rows.append({
            'Model': name.replace('_', ' ').title(),
            'Accuracy': round(m['accuracy'], 4),
            'Precision': round(m['precision'], 4),
            'Recall': round(m['recall'], 4),
            'F1': round(m['f1_score'], 4),
            'AUC': round(m['roc_auc'], 4)
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.write("**Visual Comparison**")
    chart_data = []
    for name, m in results.items():
        display_name = name.replace('_', ' ').title()
        chart_data.append({'Model': display_name, 'Metric': 'Accuracy', 'Score': m['accuracy']})
        chart_data.append({'Model': display_name, 'Metric': 'Precision', 'Score': m['precision']})
        chart_data.append({'Model': display_name, 'Metric': 'Recall', 'Score': m['recall']})
        chart_data.append({'Model': display_name, 'Metric': 'AUC', 'Score': m['roc_auc']})
    fig = px.bar(pd.DataFrame(chart_data), x='Model', y='Score', color='Metric', barmode='group')
    st.plotly_chart(fig, width='stretch')
def main():
    st.title("Logistics Delay Prediction")
    page = st.sidebar.radio("Menu", ["Dashboard", "Predictions", "Routes", "Analytics", "Models"])
    df = load_data()
    pipeline = load_pipeline()
    if df is None:
        st.warning("No data found. Generate data first.")
        return
    if page == "Dashboard":
        show_dashboard(df)
    elif page == "Predictions":
        show_predictions(pipeline)
    elif page == "Routes":
        show_routes()
    elif page == "Analytics":
        show_analytics(df)
    elif page == "Models":
        show_model_perf()

if __name__ == "__main__":
    main()
