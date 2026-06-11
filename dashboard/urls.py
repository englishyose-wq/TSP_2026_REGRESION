from django.urls import path

from .views import (
    comparison_view,
    fines_view,
    index,
    powerbi_data_csv_view,
    powerbi_data_xlsx_view,
    powerbi_excel_view,
    powerbi_view,
    regression_view,
    regression_data_xlsx_view,
)

urlpatterns = [
    path("", index, name="index"),
    path("regression/", regression_view, name="regression"),
    path("comparison/", comparison_view, name="comparison"),
    path("fines/", fines_view, name="fines"),
    path("powerbi/", powerbi_view, name="powerbi"),
    path("powerbi/excel/", powerbi_excel_view, name="powerbi_excel"),
    path("powerbi/data.csv", powerbi_data_csv_view, name="powerbi_data_csv"),
    path("powerbi/data.xlsx", powerbi_data_xlsx_view, name="powerbi_data_xlsx"),
    path("regression/data.xlsx", regression_data_xlsx_view, name="regression_data_xlsx"),
]
