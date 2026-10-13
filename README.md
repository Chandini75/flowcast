# FlowCast

Hospital flow dashboard with a Python forecasting backend.

- `static/index.html`  the dashboard (simulation, recommendations, with/without comparison)
- `forecaster.py`      the arrivals model (gradient boosting) and its data loading
- `train.py`           trains the model and prints its accuracy on held-out data
- `server.py`          FastAPI server the dashboard calls for forecasts

## Run it

```bash
pip install -r requirements.txt

# 1. Put the real Iowa ED arrivals CSV at data/uihc_ed_arrivals.csv, then train:
python train.py --data data/uihc_ed_arrivals.csv

# No file yet? This tests the pipeline with made-up data (accuracy means nothing):
python train.py --synthetic

# 2. Start the server and open http://localhost:8000
uvicorn server:app --reload --port 8000
```

The CSV needs columns `Year, Month, Day, Hour, Arrivals`.

If the server is not running, the dashboard still works with a built-in rule-based
forecast, and a note at the top says so.

## What is real and what is not

- Real: the arrivals forecast, when trained on the Iowa data. The accuracy printed by
  `train.py` and shown on the dashboard is measured on the newest 20% of the data,
  which the model never saw in training.
- Simulated: patients, beds, discharges, staffing, and the surge. The surge was tuned
  to show how the recommendations work, so its with/without result is not a real-world result.
- The model sees recent arrival counts from the dashboard. A surge much larger than
  anything in the training data will be under-predicted.

## API

- `GET  /api/metrics`   training data, dates, and held-out accuracy
- `GET  /api/profile?weekday=1`   typical arrivals per hour (Monday = 0), scaled to the demo hospital
- `POST /api/forecast`  `{weekday, hour, month, recent:[hourly counts], horizons}` returns `arrivals`, `lo`, `hi`
