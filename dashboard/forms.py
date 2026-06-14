from django import forms


class UploadForm(forms.Form):
    data_file = forms.FileField(label="Archivo CSV o Excel", required=False)
    model_type = forms.ChoiceField(
        label="Tipo de regresión",
        choices=(
            ("linear", "Lineal"),
            ("sqrt", "Raíz cuadrada"),
            ("quadratic", "Polinómica de grado 2"),
            ("sqrt_fines", "Raíz cuadrada + finos"),
            ("sqrt_log_fines", "Raíz cuadrada + log(Finos)"),
        ),
        initial="linear",
        required=False,
    )


class FinesUploadForm(forms.Form):
    data_file = forms.FileField(label="Archivo CSV o Excel", required=False)
    regression_type = forms.ChoiceField(
        label="Tipo de ajuste",
        choices=(
            ("linear", "Lineal"),
            ("logarithmic", "Logarítmica"),
        ),
        initial="linear",
        required=False,
    )
