import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter


def find_optimal_hyperparameters(file_path):
    print("=== Optimización Exhaustiva de Hiperparámetros para SVC ===")

    try:
        df = pd.read_csv(file_path)
        print("\nVista previa del dataset:")
        print(df.head())

        df.columns = df.columns.str.strip()
        print("\nColumnas del dataset:")
        print(df.columns)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return

    features = ['wind', 'precipitation']
    target = 'weather_id'

    if target not in df.columns or not all(f in df.columns for f in features):
        print("Error: Las columnas seleccionadas no están en el dataset.")
        return

    X = df[features]
    y = df[target]

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = SVC(class_weight='balanced', probability=True)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # GridSearchCV
    grid = GridSearchCV(svc, param_grid, refit=True, scoring='accuracy', cv=5, verbose=3)
    print("\nBuscando hiperparámetros óptimos...")
    grid.fit(X_train, y_train)

    print("\nMejores parámetros encontrados:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nReporte de clasificación con hiperparámetros óptimos:")
    print(classification_report(y_test, y_pred))

    return grid.best_params_


if __name__ == "__main__":
    file_path = "C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/clean_observations.csv"
    optimal_params = find_optimal_hyperparameters(file_path)
    print(f"Hiperparámetros óptimos: {optimal_params}")
