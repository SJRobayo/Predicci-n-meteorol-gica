import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from .forms import PredictionForm

# Ruta al archivo CSV
CSV_PATH = "C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/clean_observations.csv"

def load_and_train_model():
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()  # Elimina espacios alrededor de los nombres de columnas

        features = ['wind', 'precipitation']
        target = 'weather_id'

        if target not in df.columns or not all(f in df.columns for f in features):
            raise ValueError("Columnas necesarias no están en el dataset.")

        # Preprocesamiento
        X = df[features]
        y = df[target]

        # Escalado
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Aplicar SMOTE
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

        # División de datos
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Configuración del modelo
        model = SVC(
            C=100,
            gamma=0.1,
            kernel='rbf',
            class_weight='balanced',
            probability=True
        )
        model.fit(X_train, y_train)

        return model, scaler
    except Exception as e:
        raise ValueError(f"Error al cargar y entrenar el modelo: {e}")

def predict_view(request):
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extraer datos del formulario
            wind = form.cleaned_data['wind']
            precipitation = form.cleaned_data['precipitation']

            try:
                # Cargar modelo entrenado
                model, scaler = load_and_train_model()

                # Escalar los datos ingresados
                X_input = scaler.transform([[wind, precipitation]])

                # Realizar la predicción
                prediction = model.predict(X_input)

                # Responder con el resultado
                return JsonResponse({'prediction': int(prediction[0])})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
    else:
        form = PredictionForm()

    return render(request, 'base.html', {'form': form})
