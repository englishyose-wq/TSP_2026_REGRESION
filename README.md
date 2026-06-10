# Geotech Regression Dashboard

Aplicacion en Python para estimar el ángulo de fricción interna (φ) a partir de N60 mediante modelos de regresión.

## Requisitos
- Python 3.10 o superior

## Instalacion
1. Crear y activar un entorno virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso por linea de comandos

```bash
python cli.py --file data/tu_archivo.xlsx --outdir outputs
```

## Uso con Django

```bash
python manage.py runserver
```

Luego abre el navegador en la URL local que muestra Django y sube el archivo CSV o Excel.
