import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

def main():
    print("=== Clasificación del Clima con SVC ===")

    file_path = "C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/clean_observations.csv"

    try:
        df = pd.read_csv(file_path)
        print("\nVista previa del dataset:")
        print(df.head())

        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()  # Elimina espacios al inicio y final de los nombres
        print("\nColumnas del dataset:")
        print(df.columns)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return

    # Variables de interés
    features = ['date_id', 'wind', 'precipitation']
    target = 'weather_id'

    # Verificación de columnas
    if target not in df.columns or not all(f in df.columns for f in features):
        print("Error: Las columnas seleccionadas no están en el dataset.")
        return

    # Preprocesamiento
    X = df[features]
    y = df[target]

    # Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("\nDistribución inicial de clases:")
    print(Counter(y))

    try:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("\nDespués de aplicar SMOTE:")
        print("Tamaño de X:", X.shape)
        print("Tamaño de y:", y.shape)

        print("\nDistribución de clases después de SMOTE:")
        print(Counter(y))
    except Exception as e:
        print(f"Error al aplicar SMOTE: {e}")
        return

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Configuración del modelo
    optimal_model = SVC(
        C=100,
        gamma=0.1,
        kernel='rbf',
        class_weight='balanced',
        probability=True
    )

    # Entrenamiento
    print("Entrenando modelo SVC...")
    optimal_model.fit(X_train, y_train)

    # Predicciones
    y_pred = optimal_model.predict(X_test)

    # Evaluación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    matriz = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(matriz)

    # Visualización de la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=optimal_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.show()

if __name__ == "__main__":
    main()
