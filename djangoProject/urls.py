from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("__reload__/", include("django_browser_reload.urls")),
    path('admin/', admin.site.urls),
    path('index/', include('proyecto3.urls')),  # Agregar barra al final
    path('', include('proyecto3.urls')),       # Ruta para la raíz
]
