import os
import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory (ensures correct file locations)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'history.db')
data_path = os.path.join(BASE_DIR, 'clean.csv')
model_path = os.path.join(BASE_DIR, 'xgb_pca_pipeline_manual.pkl')

# Load resources
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def get_db_connection(path=db_path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            airline_code TEXT,
            origin_state TEXT,
            dest_state TEXT,
            origin_city TEXT,
            dest_city TEXT,
            scheduled_time REAL,
            air_time REAL,
            distance REAL,
            fare REAL,
            airplane_cost REAL,
            avg_plane_age REAL,
            dep_hour INTEGER,
            arr_hour INTEGER,
            dep_time_of_day TEXT,
            arr_time_of_day TEXT,
            flight_length TEXT,
            route TEXT,
            avg_delay_airline REAL,
            avg_delay_route REAL,
            route_traffic INTEGER,
            day_of_week INTEGER,
            severe_weather INTEGER,
            pred_prob REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    return conn

# Cached lookup builders
@st.cache_data
def build_airline_data(df):
    ag = df.groupby('Airline_Code').agg(
        Avg_Airline_Planes_Age=('Avg_Airline_Planes_Age', 'mean'),
        AvgDepDelayByAirline_min=('AvgDepDelayByAirline_min', 'mean'),
        Airplane_Cost_millions=('Airplane_Cost_millions', 'mean')
    )
    return ag.to_dict('index')

@st.cache_data
def build_route_data(df):
    ag = df.groupby('Route').agg(
        OriginState=('OriginState', 'first'),
        DestState=('DestState', 'first'),
        AvgDepDelayByRoute_min=('AvgDepDelayByRoute_min', 'mean'),
        RouteTraffic=('RouteTraffic', 'mean')
    )
    return ag.to_dict('index')

@st.cache_data
def build_routes_by_airline(df):
    return df.groupby('Airline_Code')['Route'].agg(lambda x: sorted(x.unique())).to_dict()

@st.cache_data
def get_airlines(df):
    return sorted(df['Airline_Code'].unique())

# Utility
def safe_to_float(val):
    try:
        return float(val)
    except:
        return np.nan

# Load pipeline and resources
df = load_data(data_path)
pipeline = load_model(model_path)
model = pipeline['model']
scaler = pipeline['scaler']
pca = pipeline['pca']
encoded_columns = pipeline['encoded_columns']
db_conn = get_db_connection(db_path)

# Precompute lookup dictionaries
airline_data = build_airline_data(df)
route_data = build_route_data(df)
routes_by_airline = build_routes_by_airline(df)
airlines = get_airlines(df)

# Cached history
# @st.cache_data(show_spinner=False)
def load_history():
    hist_df = pd.read_sql_query(
        'SELECT * FROM predictions ORDER BY timestamp DESC',
        db_conn,
        parse_dates=['timestamp']
    )

    # hist_df['pred_prob'] = hist_df['pred_prob'].apply(safe_to_float)
    hist_df.dropna(subset=['pred_prob'], inplace=True)
    return hist_df

# Streamlit App
st.title("Flight Delay Predictor")
tab1, tab2 = st.tabs(["Prediction", "History & Insights"])

with tab1:
    st.header("Make a Prediction")
    col1, col2 = st.columns(2)

    with col1:
        airline_code = st.selectbox("Airline Code", airlines)
        st.info(
            f"""
            • Avg plane age: {airline_data[airline_code]['Avg_Airline_Planes_Age']:.1f} yrs  
            • Avg delay: {airline_data[airline_code]['AvgDepDelayByAirline_min']:.1f} min  
            • Airplane cost: ${airline_data[airline_code]['Airplane_Cost_millions']:.1f}M
            """
        )
        day_of_week = st.selectbox(
            "Day of Week", 
            [(1, "Monday"), (2, "Tuesday"), (3, "Wednesday"), (4, "Thursday"),
             (5, "Friday"), (6, "Saturday"), (7, "Sunday")],
            format_func=lambda x: x[1]
        )[0]
        dep_hour = st.slider("Departure Hour", 0, 23, 12)
        dep_time_of_day = (
            "Morning" if 5 <= dep_hour < 12 else
            "Afternoon" if 12 <= dep_hour < 17 else
            "Evening" if 17 <= dep_hour < 21 else
            "Night"
        )

    with col2:
        routes = routes_by_airline.get(airline_code, [])
        route = st.selectbox("Select Route", routes)
        origin_state = route_data[route]['OriginState']
        dest_state = route_data[route]['DestState']
        st.info(
            f"""
            • Origin: {origin_state}  
            • Destination: {dest_state}  
            • Avg route delay: {route_data[route]['AvgDepDelayByRoute_min']:.1f} min  
            • Route traffic: {route_data[route]['RouteTraffic']:.0f}/day
            """
        )
        subset = df[(df['Airline_Code']==airline_code) & (df['Route']==route)]
        if not subset.empty:
            example = subset.iloc[0]
            scheduled_time = example['Scheduled_Elapsed_Time_min']
            air_time = example['AirTime_min']
            distance = example['Distance_miles']
            fare = example['Fare']
        else:
            scheduled_time, air_time, distance, fare = 120, 100, 500, 200

        distance = st.number_input("Distance (miles)", 50, 5000, int(distance))
        flight_length = (
            "IsShortFlight" if distance <= 500 else
            "IsMediumFlight" if distance <= 1500 else
            "IsLongFlight"
        )
        severe_weather = st.checkbox("Severe Weather Expected")

    arr_hour = (dep_hour + int(scheduled_time / 60)) % 24
    arr_time_of_day = (
        "Morning" if 5 <= arr_hour < 12 else
        "Afternoon" if 12 <= arr_hour < 17 else
        "Evening" if 17 <= arr_hour < 21 else
        "Night"
    )

    if st.button("Predict Delay Probability"):
        input_data = {
            'Airline_Code': airline_code,
            'Scheduled_Elapsed_Time_min': scheduled_time,
            'AirTime_min': air_time,
            'Distance_miles': distance,
            'OriginState': origin_state,
            'DestState': dest_state,
            'Origin_City': example.get('Origin_City', f"City in {origin_state}"),
            'Dest_City': example.get('Dest_City', f"City in {dest_state}"),
            'Fare': fare,
            'Airplane_Cost_millions': airline_data[airline_code]['Airplane_Cost_millions'],
            'Avg_Airline_Planes_Age': airline_data[airline_code]['Avg_Airline_Planes_Age'],
            'Dep_Hour': dep_hour,
            'Arr_Hour': arr_hour,
            'DepTimeOfDay': dep_time_of_day,
            'ArrTimeOfDay': arr_time_of_day,
            'IsShortFlight': int(flight_length=='IsShortFlight'),
            'IsMediumFlight': int(flight_length=='IsMediumFlight'),
            'IsLongFlight': int(flight_length=='IsLongFlight'),
            'Route': route,
            'AvgDepDelayByAirline_min': airline_data[airline_code]['AvgDepDelayByAirline_min'],
            'AvgDepDelayByRoute_min': route_data[route]['AvgDepDelayByRoute_min'],
            'RouteTraffic': route_data[route]['RouteTraffic'],
            'DayOfWeek': day_of_week,
            'SevereWeatherFlag': int(severe_weather)
        }

        with st.expander("View Features Used"):
            st.json(input_data)

        raw = pd.DataFrame([input_data])
        encoded = pd.get_dummies(raw)

        for col in encoded_columns:
            if col not in encoded:
                encoded[col] = 0
        encoded = encoded[encoded_columns].fillna(0)

        scaled = scaler.transform(encoded)
        scaled = np.nan_to_num(scaled)
        reduced = pca.transform(scaled)

        try:
            prob = model.predict_proba(reduced)[0][1]
            prob_str = f"{prob:.2}"
            print(prob_str, prob)
            colA, colB = st.columns([1, 2])
            with colA:
                if prob < 0.25:
                    st.success("Low delay risk")
                elif prob < 0.42:
                    st.warning("Moderate delay risk")
                else:
                    st.error("High delay risk")
                st.metric("Delay Probability", f"{prob:.2%}")
            with colB:
                risk_text = ("good chance", "moderate chance", "high chance")[min(int(prob * 2), 2)]
                st.write(f"This flight has a **{risk_text}** of delay.")
                st.write("**Key factors:**")
                factors = []
                if airline_data[airline_code]['AvgDepDelayByAirline_min'] > 15:
                    factors.append(f"Airline avg delay {airline_data[airline_code]['AvgDepDelayByAirline_min']:.1f} min")
                if route_data[route]['AvgDepDelayByRoute_min'] > 15:
                    factors.append(f"Route avg delay {route_data[route]['AvgDepDelayByRoute_min']:.1f} min")
                if severe_weather:
                    factors.append("Severe weather expected")
                if 'Night' in [dep_time_of_day, arr_time_of_day]:
                    factors.append("Night flights often delayed")
                if day_of_week in [6, 7]:
                    factors.append("Weekend flights busier")
                if not factors:
                    factors = ["No major factors"]
                for f in factors:
                    st.write(f)

            db_conn.execute('''
                INSERT INTO predictions (
                    airline_code, origin_state, dest_state, origin_city, dest_city,
                    scheduled_time, air_time, distance, fare, airplane_cost, avg_plane_age,
                    dep_hour, arr_hour, dep_time_of_day, arr_time_of_day, flight_length,
                    route, avg_delay_airline, avg_delay_route, route_traffic, day_of_week,
                    severe_weather, pred_prob, timestamp
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', [
                airline_code, origin_state, dest_state,
                input_data['Origin_City'], input_data['Dest_City'],
                scheduled_time, air_time, distance, fare,
                input_data['Airplane_Cost_millions'], input_data['Avg_Airline_Planes_Age'],
                dep_hour, arr_hour, dep_time_of_day, arr_time_of_day, flight_length,
                route, input_data['AvgDepDelayByAirline_min'], input_data['AvgDepDelayByRoute_min'],
                input_data['RouteTraffic'], day_of_week, int(severe_weather), prob_str, datetime.now().isoformat()
            ])
            db_conn.commit()

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

with tab2:
    st.header("Prediction History")
    hist = load_history()

    if hist.empty:
        st.info("No history found. Make a prediction first!")
    else:
        # --- Top metrics ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Delay Prob", f"{hist['pred_prob'].mean():.2%}")
        c2.metric("Total Predictions", len(hist))
        high = (hist['pred_prob'] > 0.42).sum()
        c3.metric("High Risk", f"{high} ({high/len(hist):.1%})")

        # --- Recent Predictions Table ---
        st.subheader("Recent Predictions")
        disp = hist[['timestamp', 'airline_code', 'route', 'dep_hour', 'severe_weather', 'pred_prob']].copy()
        disp['pred_prob'] = disp['pred_prob'].map(lambda x: f"{x:.2%}")
        disp['timestamp'] = disp['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        disp.columns = ['Timestamp', 'Airline', 'Route', 'Dep Hour', 'Severe Weather', 'Prob']
        st.dataframe(disp.head(10), use_container_width=True)

        # --- Insights Section ---
        st.subheader("Delay Category Distribution (Pie Chart)")
        dist = hist.copy()
        dist['cat'] = pd.cut(dist['pred_prob'], [0, 0.25, 0.42, 1], labels=['Low', 'Medium', 'High'])
        counts = dist['cat'].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.subheader("Avg Delay by Departure Hour (Line Chart)")
        delay_by_hour = hist.groupby('dep_hour')['pred_prob'].mean()
        fig2, ax2 = plt.subplots()
        delay_by_hour.plot(ax=ax2, marker='o')
        ax2.set_xlabel("Departure Hour")
        ax2.set_ylabel("Avg Delay Probability")
        ax2.set_title("Delay Trend Across the Day")
        st.pyplot(fig2)

        st.subheader("Delay by Flight Length Category (Box Plot)")
        if 'flight_length' in hist.columns:
            fig3, ax3 = plt.subplots()
            sns.boxplot(data=hist, x='flight_length', y='pred_prob', ax=ax3)
            ax3.set_ylabel("Delay Probability")
            ax3.set_xlabel("Flight Length")
            st.pyplot(fig3)

        st.subheader("Delay Risk by Weather Condition (Pie Chart)")
        if hist['severe_weather'].nunique() > 1:
            weather_delay = hist.groupby('severe_weather')['pred_prob'].mean()
            labels = ['No Severe Weather', 'Severe Weather']
            fig4, ax4 = plt.subplots()
            ax4.pie(weather_delay, labels=labels, autopct='%1.1f%%', startangle=90)
            ax4.axis('equal')
            st.pyplot(fig4)

        st.subheader("Route Delay Spread (Violin Plot)")
        rc = hist['route'].value_counts()
        valid_routes = rc[rc >= 10].index
        if len(valid_routes):
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            sns.violinplot(data=hist[hist['route'].isin(valid_routes)], x='route', y='pred_prob', ax=ax5)
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
            ax5.set_ylabel("Delay Probability")
            ax5.set_xlabel("Route")
            st.pyplot(fig5)

        st.subheader("Airline vs Day of Week Delay Heatmap")
        if 'day_of_week' in hist.columns:
            pivot = hist.pivot_table(index='airline_code', columns='day_of_week', values='pred_prob', aggfunc='mean')
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax6)
            ax6.set_ylabel("Airline")
            ax6.set_xlabel("Day of Week")
            st.pyplot(fig6)

        st.subheader("Filter Results")
        selected = st.multiselect("Airlines", hist['airline_code'].unique(), default=hist['airline_code'].unique()[:3])
        rng = st.slider("Probability Range", 0.0, 1.0, (0.0, 1.0), step=0.05)
        filt = hist.copy()
        if selected:
            filt = filt[filt['airline_code'].isin(selected)]
        filt = filt[(filt['pred_prob'] >= rng[0]) & (filt['pred_prob'] <= rng[1])]
        st.write(f"Showing {len(filt)} rows")
        fdisp = filt[['timestamp', 'airline_code', 'route', 'dep_hour', 'severe_weather', 'pred_prob']].copy()
        fdisp['pred_prob'] = fdisp['pred_prob'].map(lambda x: f"{x:.2%}")
        fdisp['timestamp'] = fdisp['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        fdisp.columns = ['Timestamp', 'Airline', 'Route', 'Dep Hour', 'Severe Weather', 'Prob']
        st.dataframe(fdisp, use_container_width=True)
