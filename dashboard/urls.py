from django.urls import path

from .views import comparison_view, fines_view, index, powerbi_view, regression_view

urlpatterns = [
    path("", index, name="index"),
    path("regression/", regression_view, name="regression"),
    path("comparison/", comparison_view, name="comparison"),
    path("fines/", fines_view, name="fines"),
    path("powerbi/", powerbi_view, name="powerbi"),
]
