import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "svc_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")
STATS_PATH = os.path.join(BASE_DIR, "model", "model_stats.json")

CSV_PATH = os.path.join(BASE_DIR, "csv", "clean_observations.csv")

def train_and_save_model():

    df = pd.read_csv(CSV_PATH)
    features = ['wind', 'precipitation']
    target = 'weather_id'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model = SVC(
        C=100,
        gamma=0.1,
        kernel='rbf',
        class_weight='balanced',
        probability=True
    )
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    # Crear estadísticas del modelo
    stats = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),  # Guardar la matriz de confusión como lista
        "data_size": len(y_resampled),
        "last_trained": pd.Timestamp.now().strftime('%B %Y')  # Fecha de última actualización
    }

    # Guardar estadísticas en un archivo JSON
    with open(STATS_PATH, 'w') as stats_file:
        json.dump(stats, stats_file, indent=4)

    # Guardar el modelo y el escalador
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Scaler guardado en: {SCALER_PATH}")
    print(f"Estadísticas del modelo guardadas en: {STATS_PATH}")

if __name__ == "__main__":
    train_and_save_model()
