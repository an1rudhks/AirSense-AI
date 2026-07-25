import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_data, get_features_target

def evaluate(name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2:   {r2:.4f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}

def main():
    df = load_data()
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(evaluate(name, y_test, y_pred))
        trained_models[name] = model

    best = min(results, key=lambda r: r["rmse"])
    best_model = trained_models[best["name"]]

    print(f"\nBest model: {best['name']} (RMSE: {best['rmse']:.2f}, R2: {best['r2']:.4f})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/aqi_model.pkl")
    print("Saved to models/aqi_model.pkl")

    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=X.columns)
        print("\nFeature importance:\n", importance.sort_values(ascending=False))

if __name__ == "__main__":
    main()