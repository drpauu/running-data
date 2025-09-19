# Canalització de dades de running

Ecosistema en Python per descarregar activitats de l'API de Strava, convertir els fluxos en finestres regulars de 5 segons i entrenar una xarxa convolucional temporal (TCN) amb incrustacions per atleta per predir el ritme en segons per quilòmetre.

## Requisits

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Descarrega de dades detallades

```bash
python scripts/fetch_strava.py --access-token <TOKEN>
```

* Desa `strava_streams.csv` amb les columnes `athlete_id,activity_id,ts,hr,cadence,speed_mps,grade,rpe`.
* Desa `athletes.csv` amb `athlete_id,fcmax,critical_speed_mps` (la CS és opcional).

### Paràmetres útils

* `--after/--before`: filtra per dates ISO UTC (`2024-01-01T00:00:00`).
* `--force`: sobreescriu els CSV existents.
* També pots utilitzar la variable d'entorn `STRAVA_ACCESS_TOKEN`.

## Resum de rendiment

```bash
python scripts/resum_strava.py --access-token <TOKEN>
```

* Exporta totes les activitats disponibles a `resum_activitats.csv` (pots canviar la ruta amb `--output`).
* Calcula mètriques de rendiment (quilòmetres, temps en moviment, ritme mitjà, FC mitjana) i mostra un resum general al terminal.
* Accepta els mateixos filtres de data (`--after/--before`) que l'eina de descàrrega.

## Entrenament del model

```bash
python scripts/train_tcn.py \
    --streams strava_streams.csv \
    --athletes athletes.csv \
    --epochs 30 \
    --device cuda
```

1. Converteix la velocitat a ritme i calcula senyals relatives (percentatge de FC màxima, %CS, decoupling, cadència normalitzada).
2. Re-mostra cada 5 segons i genera agregats rodants (1 i 5 minuts).
3. Crea finestres de 300 segons (60 passos) i entrena la TCN amb l'incrustació d'atleta.
4. Avalua amb MAE (s/km) sobre el conjunt de validació separat per activitat.

El millor model es guarda a `best_pace_predictor.pt` (o la ruta indicada amb `--model-out`).

## Ajustos recomanats

* Ajusta `--window` (180-480 s) i `--sample` segons la resolució dels teus fluxos.
* Si no tens Critical Speed, deixa la columna `critical_speed_mps` en blanc i el codi utilitzarà la velocitat directa.
* Per fer fine-tuning per atleta, executa el script amb dades recents i una taxa d'aprenentatge més baixa.
