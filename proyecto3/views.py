import json

from django.shortcuts import render
from .forms import PredictionForm
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "proyecto3", "model", "svc_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "proyecto3", "model", "scaler.joblib")
STATS_PATH = os.path.join(BASE_DIR, "proyecto3", "model", "model_stats.json")


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

weather_map = {
    1: {"description": "Tormenta", "image": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3J2bWprZjYya3hoY25jdWI1NDFhaTU5ZmxtbG95dzBiZTZlc3JjZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gKfyusl0PRPdTNmwnD/giphy.gif"},
    2: {"description": "Lluvia", "image": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWxvOXJteDF6dHJkejFvemtzNDc0N296cnE5ZnpkNnFhYmU1NGJkMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pNn4hlkovWAHfpLRRD/giphy.gif"},
    3: {"description": "Nublado", "image": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDNqYnNteXRmcHdtandiNThvZTlpcmZydDhwZjdseW1hcHhrNXQ2NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1TpGKApbHmkZa/giphy.gif"},
    4: {"description": "Niebla", "image": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjM2NWU2dnV3dXNncjhlZWJpN3k3bm1kZnNyM2QzOWJpbzltZ3g3ZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/W0sgn9xy8Mul3ab0mG/giphy.gif"},
    5: {"description": "Soleado", "image": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXRyNmJuazl1cmhmNTE0eTg1am03eGtoN3FsdXoycGlrY2tubWV6YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qJzZ4APiDZQuJDY7vh/giphy.gif"},
}

def predict_view(request):
    prediction = None
    weather_description = None
    image_url = None
    error = None

    # Cargar estadísticas del modelo
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as stats_file:
            model_info = json.load(stats_file)
    else:
        model_info = {
            "name": "SVC (Support Vector Machine)",
            "accuracy": "No disponible",
            "classification_report": {},
            "confusion_matrix": [],
            "data_size": "No disponible",
            "last_trained": "No disponible"
        }

    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            wind = form.cleaned_data['wind']
            precipitation = form.cleaned_data['precipitation']

            try:
                X_input = scaler.transform([[wind, precipitation]])
                weather_id = model.predict(X_input)[0]

                if weather_id in weather_map:
                    weather_description = weather_map[weather_id]["description"]
                    image_url = weather_map[weather_id]["image"]
                else:
                    weather_description = "Clima desconocido"
                    image_url = "https://example.com/default.jpg"
            except Exception as e:
                error = f"Error al realizar la predicción: {e}"
        else:
            error = "Formulario inválido."
    else:
        form = PredictionForm()

    weather_descriptions = [weather_map[i]["description"] for i in sorted(weather_map.keys())]

    return render(request, 'base.html', {
        'form': form,
        'prediction': weather_description,
        'image_url': image_url,
        'error': error,
        'model_info': model_info,
        'confusion_matrix': model_info.get('confusion_matrix', []),
        'weather_descriptions': weather_descriptions  # Pasar las descripciones al contexto
    })
