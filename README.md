# Running Data Pipeline

Herramientas para descargar actividades desde la API de Strava, preparar los
streams en ventanas de 5 segundos y entrenar una red temporal convolucional
(TCN) con embeddings por atleta para predecir el ritmo (s/km).

## Requisitos

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Descarga de datos

```bash
python scripts/fetch_strava.py --access-token <TOKEN>
```

* Guarda `strava_streams.csv` con columnas:
  `athlete_id,activity_id,ts,hr,cadence,speed_mps,grade,rpe`.
* Guarda `athletes.csv` con `athlete_id,fcmax,critical_speed_mps` (CS opcional).

### Parámetros útiles

* `--after/--before`: filtra por fechas ISO UTC (`2024-01-01T00:00:00`).
* `--force`: sobrescribe CSV existentes.
* También puedes usar la variable de entorno `STRAVA_ACCESS_TOKEN`.

## Entrenamiento del modelo

```bash
python scripts/train_tcn.py \
    --streams strava_streams.csv \
    --athletes athletes.csv \
    --epochs 30 \
    --device cuda
```

1. Convierte la velocidad a ritmo y calcula señales relativas (HR%max, %CS,
   decoupling, cadencia normalizada).
2. Re-muestrea cada 5s y crea agregados rodantes (1 y 5 min).
3. Genera ventanas de 300s (60 pasos) y entrena la TCN + embedding de atleta.
4. Evalúa con MAE (s/km) sobre el conjunto de validación separado por
   actividad.

El mejor modelo queda guardado en `best_pace_predictor.pt` (o la ruta indicada
con `--model-out`).

## Ajustes recomendados

* Ajusta `--window` (180-480 s) y `--sample` según la resolución de tus
  streams.
* Si no tienes Critical Speed, deja la columna `critical_speed_mps` en blanco.
  El código usará el ritmo directo.
* Para fine-tuning por atleta, vuelve a ejecutar el script con datos recientes
  y una tasa de aprendizaje más baja.
