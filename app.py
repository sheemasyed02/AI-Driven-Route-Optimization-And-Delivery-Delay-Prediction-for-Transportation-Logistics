import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import json
from datetime import datetime, timedelta
import joblib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from realtime_pipeline import RealtimePredictionPipeline
from optimize_routes import RouteOptimizer, Location
from realtime_api import RealtimeDataService
st.set_page_config(page_title="Logistics Dashboard", layout="wide")

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

def draw_map(locs, route=None):
    center_lat = sum(l['lat'] for l in locs) / len(locs)
    center_lon = sum(l['lon'] for l in locs) / len(locs)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    for i, loc in enumerate(locs):
        color = 'red' if i == 0 else 'blue'
        folium.Marker([loc['lat'], loc['lon']], popup=loc['name'], 
                     icon=folium.Icon(color=color)).add_to(m)
    if route:
        coords = [[locs[i]['lat'], locs[i]['lon']] for i in route]
        folium.PolyLine(coords, color='green', weight=3).add_to(m)
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
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.write("**Delays by Traffic**")
        traffic_data = df.groupby('traffic_level')['is_delayed'].mean() * 100
        fig = px.line(x=traffic_data.index, y=traffic_data.values,
                     labels={'x': 'Traffic Level', 'y': 'Delay %'}, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.write("**Vehicle Distribution**")
        veh_counts = df['vehicle_type'].value_counts()
        fig = px.pie(values=veh_counts.values, names=veh_counts.index, hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.write("**Distance vs Delay**")
        sample = df.sample(min(500, len(df)))
        fig = px.scatter(sample, x='distance_km', y='delay_minutes', color='is_delayed',
                        color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
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
    city_coords = {
        'Mumbai': (19.0760, 72.8777),
        'Delhi': (28.6139, 77.2090),
        'Bangalore': (12.9716, 77.5946),
        'Chennai': (13.0827, 80.2707),
        'Hyderabad': (17.3850, 78.4867),
        'Kolkata': (22.5726, 88.3639),
        'Pune': (18.5204, 73.8567),
        'Ahmedabad': (23.0225, 72.5714)
    }
    
    city = st.selectbox("City", list(city_coords.keys()))
    lat, lon = city_coords[city]
    cond = api.get_conditions(lat, lon)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Weather", cond['weather']['condition'])
    mc2.metric("Temp", f"{cond['weather']['temperature']}°C")
    mc3.metric("Traffic", f"{cond['traffic']['level']}/10")
    st.write("---")
    st.write("**Enter Delivery Details**")
    
    # i/p
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
    if 'locs' not in st.session_state:
        st.session_state.locs = [
            {'name': 'Warehouse (Mumbai)', 'lat': 19.0760, 'lon': 72.8777},
            {'name': 'Stop 1 (Delhi)', 'lat': 28.6139, 'lon': 77.2090},
            {'name': 'Stop 2 (Pune)', 'lat': 18.5204, 'lon': 73.8567},
            {'name': 'Stop 3 (Bangalore)', 'lat': 12.9716, 'lon': 77.5946}
        ]
    st.write("**Locations**")
    for i, loc in enumerate(st.session_state.locs):
        cols = st.columns([3, 2, 2])
        loc['name'] = cols[0].text_input(f"Name", loc['name'], key=f"n{i}")
        loc['lat'] = cols[1].number_input(f"Lat", value=loc['lat'], key=f"la{i}", format="%.4f")
        loc['lon'] = cols[2].number_input(f"Lon", value=loc['lon'], key=f"lo{i}", format="%.4f")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("+ Add Stop"):
            st.session_state.locs.append({'name': 'New Stop', 'lat': 20.0, 'lon': 77.0})
            st.rerun()
    with col2:
        algo = st.selectbox("Algorithm", ["OR-Tools", "Dijkstra", "A*"])
    if st.button("Find Best Route"):
        locs = [Location(i, l['name'], l['lat'], l['lon']) 
                for i, l in enumerate(st.session_state.locs)]
        opt = RouteOptimizer()
        algo_key = {"OR-Tools": "ortools", "Dijkstra": "dijkstra", "A*": "astar"}[algo]
        if algo_key == "ortools":
            result = opt.optimize_route(locs, algorithm='ortools', depot_idx=0, num_vehicles=1)
            total_dist = result['total_distance_km']
            route = result['routes'][0]['route']
        else:
            dist, path = opt.optimize_route(locs, algorithm=algo_key, start_idx=0, 
                                           end_idx=len(locs)-1)
            total_dist = dist
            route = path
        st.write("---")
        st.metric("Total Distance", f"{total_dist:.1f} km")
        names = [st.session_state.locs[i]['name'] for i in route]
        st.write("**Route:** " + " -> ".join(names))
        route_map = draw_map(st.session_state.locs, route)
        folium_static(route_map, width=700, height=400)

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
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Traffic-Weather Heatmap**")
    heat = df.groupby(['traffic_level', 'weather_condition'])['is_delayed'].mean() * 100
    heat = heat.unstack(fill_value=0)
    fig = px.imshow(heat, labels={'color': 'Delay %'}, color_continuous_scale='OrRd')
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)

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
