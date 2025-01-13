import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter

def entrenar_modelo(X_train, X_test, y_train, y_test, best_params):
    modelo_svr = SVR(**best_params)
    modelo_svr.fit(X_train, y_train)
    y_pred = modelo_svr.predict(X_test)
    return y_pred, modelo_svr

def calcular_matriz_confusion(y_test, y_pred, tolerancia):
    y_test_clasificado = np.floor(y_test / tolerancia)
    y_pred_clasificado = np.floor(y_pred / tolerancia)

    matriz = confusion_matrix(y_test_clasificado, y_pred_clasificado)
    return matriz, y_test_clasificado, y_pred_clasificado

def main_ejecucion():
    print("=== Ejecución del Modelo ===")

    file_path = "C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/clean_observations.csv"

    try:
        df = pd.read_csv(file_path)
        print("\nVista previa del dataset:")
        print(df.head())
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return

    print("\nColumnas disponibles:")
    print(df.columns.tolist())

    target = input("Selecciona la variable objetivo (target): ").strip()
    features = input("Selecciona las características (features), separadas por comas: ").strip().split(",")
    features = [col.strip() for col in features]

    if target not in df.columns or not all(f in df.columns for f in features):
        print("Error: Las columnas seleccionadas no están en el dataset.")
        return

    # Preprocesamiento
    X = df[features]
    y = df[target]

    # Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # SMOTE
    try:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    except Exception as e:
        print(f"Error al aplicar SMOTE: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Cargar los mejores hiperparámetros
    try:
        with open("best_params.txt", "r") as file:
            best_params = eval(file.read())
    except Exception as e:
        print(f"Error al cargar los mejores hiperparámetros: {e}")
        return

    print("Entrenando modelo con los mejores hiperparámetros...")
    y_pred, modelo = entrenar_modelo(X_train, X_test, y_train, y_test, best_params)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nResultados del modelo:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")

    # Matriz de confusión
    tolerancia = float(input("Introduce la tolerancia para clasificar los valores (e.g., 1.0): ").strip())
    matriz, y_test_clasificado, y_pred_clasificado = calcular_matriz_confusion(y_test, y_pred, tolerancia)

    print("\nMatriz de confusión (basada en la tolerancia):")
    print(matriz)

    # Visualización de la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot()
    plt.title("Matriz de Confusión (Regresión Agrupada por Tolerancia)")
    plt.show()

    print("Generando gráfico...")
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs Valores Reales")
    plt.show()

if __name__ == "__main__":
    main_ejecucion