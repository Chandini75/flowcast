"""FlowCast arrivals forecaster.

Trains a gradient boosting model that predicts how many patients arrive in the
ED in each of the next 13 hours. It uses the calendar (hour, weekday, month,
holidays) and how busy the last few hours have been compared with normal.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MAX_H = 13  # forecast this many hours ahead
TARGET_DAILY = 131.5  # patients per day at the demo hospital the dashboard simulates
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "flowcast_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


# ---------------------------------------------------------------- data loading
def load_dataset(path) -> pd.Series:
    """Read an hourly arrivals CSV (Iowa format) into a clean hourly series.

    Needs columns Year, Month, Day, Hour, Arrivals (any capitalisation).
    """
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    missing = [n for n in ("year", "month", "day", "hour", "arrivals") if n not in cols]
    if missing:
        raise ValueError(f"CSV is missing columns {missing}. Found: {list(df.columns)}")
    hour = df[cols["hour"]].astype(int)
    if hour.min() == 1 and hour.max() == 24:
        hour = hour - 1
    ts = pd.to_datetime(
        pd.DataFrame(
            {"year": df[cols["year"]], "month": df[cols["month"]], "day": df[cols["day"]], "hour": hour}
        )
    )
    s = pd.Series(df[cols["arrivals"]].astype(float).to_numpy(), index=pd.DatetimeIndex(ts))
    return clean_series(s)


def clean_series(s: pd.Series) -> pd.Series:
    s = s[~s.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).interpolate(limit=24, limit_direction="both")
    if s.isna().any():
        typ = s.groupby([s.index.dayofweek, s.index.hour]).transform("mean")
        s = s.fillna(typ)
    return s.clip(lower=0)


def make_synthetic(years: int = 3, seed: int = 0) -> pd.Series:
    """Fake hourly arrivals, only so the code can run without the real file."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=24 * 365 * years, freq="h")
    shape = np.array([3, 2.5, 2.2, 2, 2, 2.3, 3, 4, 5.5, 6.5, 7, 7.5, 7.5, 7, 7, 7.5, 8, 8.5, 8.5, 8, 7, 6, 5, 4.0])
    wd_f = np.array([1.1, 1.0, 1.0, 1.0, 1.0, 0.95, 0.95])
    season = 1 + 0.08 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 30) / 365)
    days = len(idx) // 24
    level = np.zeros(days)
    for i in range(1, days):
        level[i] = 0.6 * level[i - 1] + rng.normal(0, 0.1)
    lam = shape[idx.hour] * wd_f[idx.dayofweek] * season * np.exp(np.repeat(level, 24)[: len(idx)])
    lam = lam * np.where(holiday_flags(idx) == 1, 0.85, 1.0)
    return pd.Series(rng.poisson(lam).astype(float), index=idx)


# ------------------------------------------------------------------- features
def holiday_flags(idx: pd.DatetimeIndex) -> np.ndarray:
    m, d, wd = idx.month, idx.day, idx.dayofweek
    h = (
        ((m == 12) & (d >= 24) & (d <= 26))
        | ((m == 1) & (d == 1))
        | ((m == 7) & (d == 4))
        | ((m == 11) & (wd == 3) & (d >= 22) & (d <= 28))
    )
    return np.asarray(h, dtype=int)


def typical_table(a, wd, hr, upto) -> np.ndarray:
    """Average arrivals for each weekday and hour, using only data before `upto`."""
    df = pd.DataFrame({"a": a[:upto], "wd": wd[:upto], "hr": hr[:upto]})
    tab = np.full((7, 24), np.nan)
    for (d, h), v in df.groupby(["wd", "hr"])["a"].mean().items():
        tab[d, h] = v
    return np.where(np.isnan(tab), a[:upto].mean(), tab)


def rolling_mean(x, w):
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def build_matrix(a, wd, hr, mo, hol, tab, origins, horizons):
    """One row per (forecast origin, horizon). Only uses data up to the origin."""
    T = tab[wd, hr]
    r6, r24 = rolling_mean(a, 6), rolling_mean(a, 24)
    t6 = np.maximum(rolling_mean(T, 6), 1e-6)
    parts, ys, hs = [], [], []
    for h in horizons:
        t = origins + h
        parts.append(
            np.column_stack(
                [np.full(len(origins), h), hr[t], wd[t], mo[t], hol[t], T[t],
                 r6[origins], r24[origins], a[origins], r6[origins] / t6[origins]]
            )
        )
        ys.append(a[t])
        hs.append(np.full(len(origins), h))
    return np.vstack(parts), np.concatenate(ys), np.concatenate(hs)


def new_model():
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=250, learning_rate=0.08, max_leaf_nodes=24,
        min_samples_leaf=100, l2_regularization=1.0, random_state=0,
    )


# ------------------------------------------------------------------- training
def train(series: pd.Series, source: str, synthetic: bool = False, refit: bool = True) -> dict:
    a = series.to_numpy(float)
    idx = series.index
    n = len(a)
    if n < 24 * 60:
        raise ValueError("Need at least 60 days of hourly data.")
    wd, hr, mo, hol = idx.dayofweek.to_numpy(), idx.hour.to_numpy(), idx.month.to_numpy(), holiday_flags(idx)

    split = int(n * 0.8)  # oldest 80% to train, newest 20% to test
    tab_tr = typical_table(a, wd, hr, split)
    Xtr, ytr, _ = build_matrix(a, wd, hr, mo, hol, tab_tr, np.arange(23, split - MAX_H), range(1, MAX_H + 1))
    Xte, yte, hte = build_matrix(a, wd, hr, mo, hol, tab_tr, np.arange(split, n - MAX_H), range(1, MAX_H + 1))
    model = new_model().fit(Xtr, ytr)
    pred = model.predict(Xte)
    typ_pred, pers_pred = Xte[:, 5], Xte[:, 6]

    def mae(p, mask=None):
        e = np.abs(yte - p)
        return float(e.mean() if mask is None else e[mask].mean())

    by_h = {}
    for h in (1, 3, 6, 12):
        m = hte == h
        by_h[str(h)] = {
            "model": round(mae(pred, m), 3),
            "typical_hour": round(mae(typ_pred, m), 3),
            "last_6h_average": round(mae(pers_pred, m), 3),
        }
    mae_model, mae_typ = mae(pred), mae(typ_pred)
    resid_std = [float(np.std((yte - pred)[hte == h])) for h in range(1, MAX_H + 1)]

    if refit:
        tab = typical_table(a, wd, hr, n)
        X, y, _ = build_matrix(a, wd, hr, mo, hol, tab, np.arange(23, n - MAX_H), range(1, MAX_H + 1))
        model = new_model().fit(X, y)
    else:
        tab = tab_tr

    daily = float(a.mean() * 24)
    metrics = {
        "source": source,
        "synthetic": synthetic,
        "rows": int(n),
        "date_from": str(idx[0].date()),
        "date_to": str(idx[-1].date()),
        "test_hours": int(n - split),
        "unit": "patients per hour, original dataset",
        "mean_arrivals_per_hour": round(float(a.mean()), 3),
        "mae_model": round(mae_model, 3),
        "mae_typical_hour": round(mae_typ, 3),
        "mae_last_6h_average": round(mae(pers_pred), 3),
        "improvement_pct": round((mae_typ - mae_model) / mae_typ * 100, 1),
        "by_horizon": by_h,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "typ": tab.tolist(), "scale": TARGET_DAILY / daily,
         "resid_std": resid_std, "metrics": metrics},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


# -------------------------------------------------------------------- serving
def _shift(weekday: int, hour: int, delta: int):
    total = weekday * 24 + hour + delta
    return (total // 24) % 7, total % 24


class Forecaster:
    def __init__(self, path=MODEL_PATH):
        b = joblib.load(path)
        self.model, self.typ, self.scale = b["model"], b["typ"], b["scale"]
        self.resid_std, self.metrics = b["resid_std"], b["metrics"]

    def profile(self, weekday: int):
        """Typical arrivals per hour for one weekday, scaled to the demo hospital."""
        return [round(self.typ[weekday][h] * self.scale, 3) for h in range(24)]

    def predict(self, weekday, hour, month, recent, horizons=MAX_H):
        """Forecast arrivals for the hours after `hour`.

        `recent` = hourly arrivals up to and including `hour`, oldest first,
        in the dashboard's own units (they are rescaled internally).
        """
        horizons = max(1, min(int(horizons), MAX_H))
        r = np.asarray(recent, float) / self.scale
        r = r[np.isfinite(r)]
        if r.size == 0:
            r = np.array([self.typ[weekday][hour]])
        r6, r24, last = r[-6:].mean(), r[-24:].mean(), r[-1]
        t6 = np.mean([self.typ[w][h] for w, h in (_shift(weekday, hour, -i) for i in range(6))])
        ratio6 = r6 / max(t6, 1e-6)
        rows = []
        for h in range(1, horizons + 1):
            w, hh = _shift(weekday, hour, h)
            rows.append([h, hh, w, month, 0, self.typ[w][hh], r6, r24, last, ratio6])
        p = self.model.predict(np.array(rows, float)) * self.scale
        sd = np.array(self.resid_std[:horizons]) * self.scale
        return {
            "arrivals": [round(float(x), 3) for x in p],
            "lo": [round(float(x), 3) for x in np.maximum(0, p - 1.28 * sd)],
            "hi": [round(float(x), 3) for x in p + 1.28 * sd],
        }
