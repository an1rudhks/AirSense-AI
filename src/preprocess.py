import pandas as pd

FEATURES = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
TARGET = "AQI"

# Official CPCB National AQI breakpoint tables: (band_low, band_high, index_low, index_high)
PM25_BP = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,380,401,500)]
PM10_BP = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,510,401,500)]
NO2_BP  = [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,500,401,500)]
SO2_BP  = [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2100,401,500)]
CO_BP   = [(0,1.0,0,50),(1.1,2.0,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)]
O3_BP   = [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)]

def _sub_index(val, breakpoints):
    for blo, bhi, ilo, ihi in breakpoints:
        if blo <= val <= bhi:
            return (ihi - ilo) / (bhi - blo) * (val - blo) + ilo
    if val > breakpoints[-1][1]:
        return 500.0
    return 0.0

def compute_cpcb_aqi(row):
    """Official CPCB National AQI: the maximum of all pollutant sub-indices.
    CO in this dataset is in ug/m3, so it is converted to mg/m3 to match
    the official CO breakpoint table."""
    co_mgm3 = row["CO"] / 1000.0
    subs = [
        _sub_index(row["PM2.5"], PM25_BP),
        _sub_index(row["PM10"], PM10_BP),
        _sub_index(row["NO2"], NO2_BP),
        _sub_index(row["SO2"], SO2_BP),
        _sub_index(co_mgm3, CO_BP),
        _sub_index(row["O3"], O3_BP),
    ]
    return max(subs)

def load_data(path="data/cleaned_air_quality.csv"):
    """Load and clean the raw AQI dataset, recomputing AQI using the
    official CPCB formula."""
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df = df.dropna(subset=FEATURES)
    for col in FEATURES:
        df = df[df[col] >= 0]

    df["AQI"] = df.apply(compute_cpcb_aqi, axis=1)

    return df

def get_features_target(df):
    X = df[FEATURES]
    y = df[TARGET]
    return X, y

if __name__ == "__main__":
    df = load_data()
    print("Shape after cleaning:", df.shape)
    X, y = get_features_target(df)
    print("\nAQI range (official CPCB formula):", y.min(), "to", y.max())
    print("\nSample:\n", df[FEATURES + ["AQI"]].head())

def predict_aqi_official(pm25, pm10, co, no2, so2, o3):
    """Compute the official CPCB AQI directly from pollutant values.
    This is an exact calculation (not a statistical model), so it always
    matches the standard CPCB formula precisely."""
    row = {"PM2.5": pm25, "PM10": pm10, "CO": co, "NO2": no2, "SO2": so2, "O3": o3}
    return compute_cpcb_aqi(row)