from django import forms

class PredictionForm(forms.Form):
    wind = forms.FloatField(label="Wind Speed", required=True)
    precipitation = forms.FloatField(label="Precipitation", required=True)
