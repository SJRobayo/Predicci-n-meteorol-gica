### Script 1: Búsqueda de Hiperparámetros ###

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter

def buscar_mejores_hiperparametros(X, y):
    print("Buscando mejores hiperparámetros usando RandomizedSearchCV...")

    parametros = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.logspace(-3, 3, 10),
        'epsilon': np.linspace(0.01, 1, 10),
        'gamma': ['scale', 'auto']
    }

    random_search = RandomizedSearchCV(
        estimator=SVR(),
        param_distributions=parametros,
        n_iter=20,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X, y)

    print("\nMejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    print("\nMejor puntuación obtenida (MSE):")
    print(-random_search.best_score_)  # Negativo porque usamos "neg_mean_squared_error"

    return random_search.best_params_

def main_busqueda():
    print("=== Búsqueda de Hiperparámetros ===")

    file_path = "D:\Documentos\p3\proyecto3\csv\clean_observations.csv"

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
    y_clasificado = np.floor(y / 1.0)
    print("\nDistribución inicial de clases:")
    print(Counter(y_clasificado))

    try:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("\nDespués de aplicar SMOTE:")
        print("Tamaño de X:", X.shape)
        print("Tamaño de y:", y.shape)

        y_clasificado_smote = np.floor(y / 1.0)
        print("\nDistribución de clases después de SMOTE:")
        print(Counter(y_clasificado_smote))
    except Exception as e:
        print(f"Error al aplicar SMOTE: {e}")
        return

    best_params = buscar_mejores_hiperparametros(X, y)

    # Guardar los mejores hiperparámetros
    with open("best_params.txt", "w") as file:
        file.write(str(best_params))

    print("Mejores hiperparámetros guardados en 'best_params.txt'.")

if __name__ == "__main__":
    main_busqueda()