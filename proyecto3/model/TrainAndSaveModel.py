import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib

# Obtener el directorio base de tu proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas para guardar el modelo y el escalador
MODEL_PATH = os.path.join(BASE_DIR, "model", "svc_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")

# Ruta al archivo CSV
CSV_PATH = os.path.join(BASE_DIR, "csv", "clean_observations.csv")

def train_and_save_model():
    """
    Entrena el modelo SVC y lo guarda junto con el escalador.
    """
    # Cargar datos
    df = pd.read_csv(CSV_PATH)
    features = ['wind', 'precipitation']
    target = 'weather_id'

    X = df[features]
    y = df[target]

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balancear las clases con SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Dividir datos en entrenamiento y prueba
    X_train, _, y_train, _ = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = SVC(
        C=100,
        gamma=0.1,
        kernel='rbf',
        class_weight='balanced',
        probability=True
    )
    model.fit(X_train, y_train)

    # Guardar el modelo y el escalador
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Scaler guardado en: {SCALER_PATH}")

if __name__ == "__main__":
    train_and_save_model()
