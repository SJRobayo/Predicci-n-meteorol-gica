# views.py
from django.http import HttpResponse
from django.shortcuts import render

def homepage_view(request):
    return render(request, 'base.html')
