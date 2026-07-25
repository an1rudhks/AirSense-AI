import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu

from preprocess import load_data, FEATURES, predict_aqi_official

st.set_page_config(page_title="AirSense-AI", page_icon="🌎", layout="wide")

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO2": "µg/m³",
    "SO2": "µg/m³",
    "O3": "µg/m³",
    "CO": "µg/m³",
}

def with_unit(feat):
    return f"{feat} ({UNITS[feat]})"

# ---------- Custom CSS ----------
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #1c1f26;
    border-radius: 10px;
    padding: 15px 10px;
    border: 1px solid #2a2e37;
}
.quick-stat-box {
    background-color: #1c1f26;
    border-radius: 10px;
    padding: 12px 15px;
    margin-bottom: 10px;
    border: 1px solid #2a2e37;
}
</style>
""", unsafe_allow_html=True)

# ---------- Cached loaders ----------
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    return joblib.load("models/aqi_model.pkl")

df = get_data()
model = get_model()

# ---------- AQI category helper: OFFICIAL INDIAN CPCB SCALE ----------
def aqi_category(aqi):
    if aqi <= 50:
        return ("Good", "#009865",
                "Minimal impact. Air quality is considered satisfactory, and air pollution poses little or no risk.")
    elif aqi <= 100:
        return ("Satisfactory", "#a3c853",
                "Minor breathing discomfort to sensitive people (asthma, lung/heart disease, children, elderly).")
    elif aqi <= 200:
        return ("Moderate", "#ffd400",
                "Breathing discomfort to people with lung disease such as asthma, and discomfort to people with heart disease, children and older adults.")
    elif aqi <= 300:
        return ("Poor", "#ff7e00",
                "Breathing discomfort to most people on prolonged exposure. Limit outdoor exertion, especially for sensitive groups.")
    elif aqi <= 400:
        return ("Very Poor", "#ff0000",
                "Respiratory illness on prolonged exposure. Avoid outdoor activity, particularly for children, elderly, and those with respiratory/heart conditions.")
    else:
        return ("Severe", "#7e0023",
                "Affects healthy people and seriously impacts those with existing diseases. Avoid all outdoor physical activity; stay indoors.")

# ---------- Sidebar: Quick Stats ----------
with st.sidebar:
    st.markdown("## 🌎 AirSense-AI")
    st.markdown("### 📌 Quick Stats")

    avg_aqi_overall = df["AQI"].mean()
    city_avgs = df.groupby("City")["AQI"].mean()
    most_polluted = city_avgs.idxmax()
    cleanest = city_avgs.idxmin()

    st.markdown(f"""
    <div class="quick-stat-box">📊 Avg AQI (2025): <b>{avg_aqi_overall:.0f}</b></div>
    <div class="quick-stat-box">🔴 Most Polluted: <b>{most_polluted}</b> ({city_avgs[most_polluted]:.0f})</div>
    <div class="quick-stat-box">🟢 Cleanest City: <b>{cleanest}</b> ({city_avgs[cleanest]:.0f})</div>
    <div class="quick-stat-box">🤖 Model Accuracy: <b>R² 0.99</b></div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Data: 4 Indian cities, Jan-Dec 2025 · Scale: CPCB National AQI")

# ---------- Top navigation ----------
page = option_menu(
    menu_title=None,
    options=["Dashboard", "Health Advisory", "Predict AQI", "Input Guide"],
    icons=["bar-chart", "heart-pulse", "cpu", "book"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0e1117"},
        "icon": {"color": "#00c853", "font-size": "16px"},
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "padding": "10px"},
        "nav-link-selected": {"background-color": "#1c1f26", "color": "#00c853"},
    }
)

# =========================================================
# PAGE 1: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.title("📊 AQI Dashboard — 2025")
    st.caption("Overview of air quality across Chennai, Delhi, Kolkata, and Mumbai — Indian CPCB AQI scale")

    col1, col2, col3, col4 = st.columns(4)
    for col, city in zip([col1, col2, col3, col4], df["City"].unique()):
        avg_aqi = df[df["City"] == city]["AQI"].mean()
        cat, color, _ = aqi_category(avg_aqi)
        col.metric(city, f"{avg_aqi:.0f}", cat)

    st.divider()

    st.subheader("Average AQI by City")
    avg_by_city = df.groupby("City")["AQI"].mean().reset_index().sort_values("AQI", ascending=False)
    fig1 = px.bar(avg_by_city, x="City", y="AQI", color="City", text_auto=".0f")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Monthly Average AQI")
    trend_city = st.selectbox("Select city", ["All Cities"] + list(df["City"].unique()), key="trend_city")

    monthly = df.copy()
    monthly["MonthName"] = monthly["Datetime"].dt.strftime("%b")
    monthly["MonthName"] = pd.Categorical(monthly["MonthName"], categories=MONTH_ORDER, ordered=True)

    if trend_city == "All Cities":
        monthly_avg = monthly.groupby(["MonthName", "City"], observed=True)["AQI"].mean().reset_index()
        fig2 = px.bar(monthly_avg.sort_values("MonthName"), x="MonthName", y="AQI",
                      color="City", barmode="group", labels={"MonthName": "Month"})
    else:
        city_monthly = monthly[monthly["City"] == trend_city].groupby("MonthName", observed=True)["AQI"].mean().reset_index()
        fig2 = px.bar(city_monthly.sort_values("MonthName"), x="MonthName", y="AQI",
                      text_auto=".0f", labels={"MonthName": "Month"},
                      color_discrete_sequence=["#00c853"])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Pollutant Comparison Across Cities")
    pollutant = st.selectbox("Choose a pollutant", FEATURES, format_func=with_unit)
    comp = df.groupby("City")[pollutant].agg(["mean", "std"]).reset_index()
    fig3 = px.bar(comp, x="City", y="mean", error_y="std", color="City", text_auto=".1f",
                  labels={"mean": with_unit(pollutant)})
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"Bars show average {pollutant} ({UNITS[pollutant]}) per city. Error bars show day-to-day variability.")

    st.subheader("Monthly Pollutant Trend — Single City")
    city_choice = st.selectbox("Choose a city", df["City"].unique(), key="pollutant_trend_city")
    city_df = df[df["City"] == city_choice].copy()
    city_df["MonthName"] = city_df["Datetime"].dt.strftime("%b")
    city_df["MonthName"] = pd.Categorical(city_df["MonthName"], categories=MONTH_ORDER, ordered=True)
    city_monthly_pollutants = city_df.groupby("MonthName", observed=True)[FEATURES].mean().reset_index().sort_values("MonthName")

    fig4 = make_subplots(rows=3, cols=2, subplot_titles=[with_unit(f) for f in FEATURES])
    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    colors = ["#00c853", "#2196f3", "#ff5252", "#ffab00", "#7c4dff", "#00bcd4"]
    for (r, c), feat, color in zip(positions, FEATURES, colors):
        fig4.add_trace(
            go.Bar(x=city_monthly_pollutants["MonthName"], y=city_monthly_pollutants[feat],
                   name=feat, marker_color=color),
            row=r, col=c
        )
    fig4.update_layout(height=700, showlegend=False, title_text=f"Monthly Average Pollutant Levels — {city_choice}")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Each pollutant is shown on its own scale and unit, since concentrations vary widely.")

# =========================================================
# PAGE 2: HEALTH ADVISORY
# =========================================================
elif page == "Health Advisory":
    st.title("🩺 Health Advisory")
    st.caption("Indian National AQI (CPCB) scale — what an AQI value means for your health")

    aqi_input = st.slider("Select an AQI value", 0, 500, 100)
    cat, color, advice = aqi_category(aqi_input)

    st.markdown(
        f"""
        <div style="background-color:{color};padding:20px;border-radius:10px;">
        <h2 style="color:black;">Category: {cat}</h2>
        <p style="color:black;font-size:16px;">{advice}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("All AQI Categories (CPCB)")
    ranges = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    for low, high in ranges:
        cat, color, advice = aqi_category(low)
        st.markdown(
            f"""
            <div style="background-color:{color};padding:12px;border-radius:8px;margin-bottom:8px;">
            <b style="color:black;">{low}-{high} — {cat}</b><br>
            <span style="color:black;">{advice}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# PAGE 3: PREDICT AQI
# =========================================================
elif page == "Predict AQI":
    st.title("🔮 Predict AQI from Pollutant Values")
    st.caption("Enter pollutant concentrations to get a predicted AQI (Indian CPCB scale)")

    presets = {
        "🟢 Good": {"PM2.5": 15.9, "PM10": 22.0, "CO": 320.6, "NO2": 9.8, "SO2": 9.3, "O3": 47.9},
        "🟡 Satisfactory": {"PM2.5": 20.7, "PM10": 30.6, "CO": 323.2, "NO2": 10.8, "SO2": 10.1, "O3": 73.4},
        "🟠 Moderate": {"PM2.5": 50.0, "PM10": 61.3, "CO": 548.7, "NO2": 20.6, "SO2": 23.3, "O3": 116.3},
        "🔴 Poor": {"PM2.5": 94.6, "PM10": 116.4, "CO": 786.6, "NO2": 28.6, "SO2": 29.8, "O3": 103.5},
        "🟣 Very Poor": {"PM2.5": 140.6, "PM10": 158.0, "CO": 989.5, "NO2": 36.1, "SO2": 38.5, "O3": 100.4},
        "🟤 Severe": {"PM2.5": 101.8, "PM10": 512.3, "CO": 523.4, "NO2": 18.7, "SO2": 20.6, "O3": 122.0},
    }

    st.write("Load an example:")
    preset_cols = st.columns(3)
    for i, (label, values) in enumerate(presets.items()):
        if preset_cols[i % 3].button(label, use_container_width=True):
            for feat, val in values.items():
                st.session_state[f"input_{feat}"] = val

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input(with_unit("PM2.5"), min_value=0.0, value=50.0, step=1.0, key="input_PM2.5")
        pm10 = st.number_input(with_unit("PM10"), min_value=0.0, value=80.0, step=1.0, key="input_PM10")
        co = st.number_input(with_unit("CO"), min_value=0.0, value=300.0, step=1.0, key="input_CO")
    with col2:
        no2 = st.number_input(with_unit("NO2"), min_value=0.0, value=10.0, step=1.0, key="input_NO2")
        so2 = st.number_input(with_unit("SO2"), min_value=0.0, value=8.0, step=1.0, key="input_SO2")
        o3 = st.number_input(with_unit("O3"), min_value=0.0, value=100.0, step=1.0, key="input_O3")

    if st.button("Predict AQI", type="primary"):
        pred = predict_aqi_official(pm25, pm10, co, no2, so2, o3)
        cat, color, advice = aqi_category(pred)

        st.markdown(
            f"""
            <div style="background-color:{color};padding:20px;border-radius:10px;">
            <h1 style="color:black;">Predicted AQI: {pred:.0f}</h1>
            <h3 style="color:black;">{cat}</h3>
            <p style="color:black;">{advice}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Calculated using the official CPCB formula (exact, not a statistical estimate).")

# =========================================================
# PAGE 4: INPUT GUIDE
# =========================================================
elif page == "Input Guide":
    st.title("📖 Input Guide")
    st.caption("Official CPCB breakpoint ranges for each pollutant — use these when entering values on the Predict page to target a specific AQI category")

    descriptions = {
        "PM2.5": "Fine particulate matter (≤2.5 micrometers). Often the dominant driver of AQI in Indian cities — mainly from vehicles, construction dust, and burning.",
        "PM10": "Coarser particulate matter (≤10 micrometers), from dust and construction.",
        "CO": "Carbon monoxide, mainly from vehicle exhaust and incomplete combustion.",
        "NO2": "Nitrogen dioxide, from vehicles and industrial emissions.",
        "SO2": "Sulfur dioxide, mainly from burning fossil fuels and industrial processes.",
        "O3": "Ground-level ozone, formed by chemical reactions between pollutants in sunlight.",
    }

    # Official CPCB breakpoint tables, expressed in this app's display units
    # (CO breakpoints are officially in mg/m3, converted x1000 to match ug/m3 display)
    CPCB_BANDS = {
        "PM2.5": [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 380)],
        "PM10":  [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 510)],
        "NO2":   [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 500)],
        "SO2":   [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 2100)],
        "CO":    [(0, 1000), (1100, 2000), (2100, 10000), (10100, 17000), (17100, 34000), (34100, 50000)],
        "O3":    [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 1000)],
    }

    CATEGORY_LABELS = ["🟢 Good", "🟡 Satisfactory", "🟠 Moderate", "🔴 Poor", "🟣 Very Poor", "🟤 Severe"]
    CATEGORY_COLORS = ["#009865", "#a3c853", "#ffd400", "#ff7e00", "#ff0000", "#7e0023"]

    st.info("These ranges come directly from the official CPCB National AQI formula — entering a value in a given band on the Predict page will push the AQI toward that category (the overall AQI is always driven by whichever single pollutant is worst).")

    for pollutant in FEATURES:
        st.markdown(f"### {pollutant} ({UNITS[pollutant]})")
        st.write(descriptions[pollutant])

        cols = st.columns(6)
        bands = CPCB_BANDS[pollutant]
        for col, (lo, hi), label, color in zip(cols, bands, CATEGORY_LABELS, CATEGORY_COLORS):
            range_text = f"{lo}–{hi}" if (lo, hi) != bands[-1] else f"> {lo}"
            col.markdown(
                f"""
                <div style="background-color:{color};padding:8px 4px;border-radius:6px;text-align:center;">
                <span style="color:black;font-size:11px;font-weight:600;">{label}</span><br>
                <span style="color:black;font-size:13px;">{range_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.divider()

    with st.expander("See this dataset's actual observed range per pollutant"):
        stats = df[FEATURES].describe().T[["min", "25%", "50%", "75%", "max"]]
        stats.columns = ["Min", "25th percentile", "Median", "75th percentile", "Max"]
        stats.index = [with_unit(f) for f in stats.index]
        st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)