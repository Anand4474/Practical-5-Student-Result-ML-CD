from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
MODEL_PATH = Path("student_result_model.pkl")

def make_demo_data(n=500, seed=42):
    # Deterministic synthetic data for this classroom practical.
    df = pd.DataFrame({
        "attendance": [],
        "internal_marks": [],
        "assignment_marks": [],
        "previous_score": [],
    })
    rng = __import__("numpy").random.default_rng(seed)
    df = pd.DataFrame({
        "attendance": rng.integers(35, 101, size=n),
        "internal_marks": rng.integers(10, 101, size=n),
        "assignment_marks": rng.integers(10, 101, size=n),
        "previous_score": rng.integers(10, 101, size=n),
    })
    weighted_score = (
        0.20 * df["attendance"] +
        0.30 * df["internal_marks"] +
        0.20 * df["assignment_marks"] +
        0.30 * df["previous_score"]
    )
    df["result"] = (weighted_score >= 50).astype(int)
    return df

def main():
    data = make_demo_data()
    data.to_csv("student_results.csv", index=False)
    X = data[FEATURES]
    y = data["result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "accuracy": round(accuracy, 4),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(json.dumps(metrics, indent=2))
    print("Model and metrics generated successfully.")

if __name__ == "__main__":
    main()
