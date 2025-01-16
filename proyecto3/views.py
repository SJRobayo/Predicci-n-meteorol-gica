from django.shortcuts import render
from .forms import PredictionForm
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Ruta al archivo CSV
CSV_PATH = "C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/clean_observations.csv"

def load_and_train_model():
    """
    Cargar datos, preprocesarlos, y entrenar el modelo.
    """
    df = pd.read_csv(CSV_PATH)
    features = ['wind', 'precipitation']
    target = 'weather_id'
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, _, y_train, _ = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model = SVC(
        C=100,
        gamma=0.1,
        kernel='rbf',
        class_weight='balanced',
        probability=True
    )
    model.fit(X_train, y_train)
    return model, scaler

def predict_view(request):
    """
    Vista para manejar el formulario y generar una predicción dinámica.
    """
    prediction = None
    weather_description = None
    image_url = None
    error = None

    weather_map = {
        1: {"description": "Tormenta", "image": "storm.gif"},
        2: {"description": "Lluvia", "image": "https://tenor.com/view/worried-scared-oh-no-stop-it-fearful-gif-12534009"},
        3: {"description": "Nublado", "image": "https://tenor.com/view/worried-scared-oh-no-stop-it-fearful-gif-12534009"},
        4: {"description": "Niebla", "image": "https://tenor.com/view/worried-scared-oh-no-stop-it-fearful-gif-12534009"},
        5: {"description": "Soleado", "image": "https://tenor.com/view/worried-scared-oh-no-stop-it-fearful-gif-12534009"},
    }

    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            wind = form.cleaned_data['wind']
            precipitation = form.cleaned_data['precipitation']

            try:
                model, scaler = load_and_train_model()
                X_input = scaler.transform([[wind, precipitation]])
                prediction = model.predict(X_input)[0]

                if prediction in weather_map:
                    weather_description = weather_map[prediction]["description"]
                    image_url = weather_map[prediction]["image"]
                else:
                    weather_description = "Clima desconocido"
                    image_url = "https://example.com/default.jpg"
            except Exception as e:
                error = f"Error al realizar la predicción: {e}"
        else:
            error = "Formulario inválido."
    else:
        form = PredictionForm()

    return render(request, 'base.html', {
        'form': form,
        'prediction': weather_description,
        'image_url': image_url,
        'error': error
    })
