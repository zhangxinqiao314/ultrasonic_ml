"""SQLite-backed helpers for ultrasonic thermal-gradient analysis.

The store keeps scalar values in normalized SQLite tables and waveform-like
values as NumPy arrays serialized in SQLite BLOBs. SQL is used for joins,
filters, updates, and grouped statistics; NumPy is used only for signal
processing and downstream plotting or machine-learning boundaries.
"""

from __future__ import annotations

import csv
import os
import pickle
import sqlite3
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

import pickleJar as pj


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id INTEGER PRIMARY KEY,
    time_collected REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scalars (
    measurement_id INTEGER NOT NULL REFERENCES measurements(measurement_id),
    name TEXT NOT NULL,
    value REAL,
    text_value TEXT,
    PRIMARY KEY (measurement_id, name)
);
CREATE TABLE IF NOT EXISTS arrays (
    measurement_id INTEGER NOT NULL REFERENCES measurements(measurement_id),
    name TEXT NOT NULL,
    value BLOB NOT NULL,
    PRIMARY KEY (measurement_id, name)
);
CREATE INDEX IF NOT EXISTS idx_scalars_name_value ON scalars(name, value);
CREATE INDEX IF NOT EXISTS idx_measurements_time ON measurements(time_collected);
"""


def _pack(value: np.ndarray) -> sqlite3.Binary:
    return sqlite3.Binary(pickle.dumps(np.asarray(value), protocol=pickle.HIGHEST_PROTOCOL))


def _unpack(value: bytes) -> np.ndarray:
    return np.asarray(pickle.loads(value))


def _timestamp(value: str) -> float:
    return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p").timestamp()


def _read_controller_csv(paths: Sequence[str], usecols: Sequence[str] | None = None) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            for row in reader:
                if usecols is not None:
                    row = {key: row.get(key) for key in usecols}
                if row.get("Time"):
                    row["Time_str"] = row["Time"]
                    row["Time"] = _timestamp(row["Time"]) + float(row.get("Milliseconds") or 0) / 1000
                    row.pop("Milliseconds", None)
                    rows.append(row)
    return sorted(rows, key=lambda row: row["Time"])


def _nearest(rows: Sequence[dict], times: Sequence[float], timestamp: float) -> dict:
    index = bisect_left(times, timestamp)
    if index == 0:
        return rows[0]
    if index == len(rows):
        return rows[-1]
    before, after = rows[index - 1], rows[index]
    return before if timestamp - before["Time"] <= after["Time"] - timestamp else after


def _coerce_scalar(value):
    if value is None or value == "":
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, str(value)


class SQLiteGradientStore:
    """SQLite data store used by the gradient-analysis workflow."""

    def __init__(self, database_path: str | os.PathLike, connection: sqlite3.Connection | None = None):
        self.database_path = str(database_path)
        self.connection = connection or sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def set_metadata(self, key: str, value) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, sqlite3.Binary(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))),
        )
        self.connection.commit()

    def get_metadata(self, key: str, default=None):
        row = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return default if row is None else pickle.loads(row[0])

    def add_measurement(self, measurement_id: int, time_collected: float, scalars: dict | None = None, arrays: dict | None = None) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO measurements(measurement_id, time_collected) VALUES (?, ?)",
            (measurement_id, float(time_collected)),
        )
        for name, value in (scalars or {}).items():
            self.set_scalar(measurement_id, name, value, commit=False)
        for name, value in (arrays or {}).items():
            self.set_array(measurement_id, name, value, commit=False)
        self.connection.commit()

    def set_scalar(self, measurement_id: int, name: str, value, commit: bool = True) -> None:
        numeric, text = _coerce_scalar(value)
        self.connection.execute(
            """INSERT OR REPLACE INTO scalars(measurement_id, name, value, text_value)
               VALUES (?, ?, ?, ?)""",
            (measurement_id, name, numeric, text),
        )
        if commit:
            self.connection.commit()

    def set_array(self, measurement_id: int, name: str, value: np.ndarray, commit: bool = True) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO arrays(measurement_id, name, value) VALUES (?, ?, ?)",
            (measurement_id, name, _pack(value)),
        )
        if commit:
            self.connection.commit()

    def set_scalar_sql(self, name: str, expression: str) -> None:
        self.connection.execute(
            f"""INSERT OR REPLACE INTO scalars(measurement_id, name, value)
                SELECT measurement_id, ?, {expression} FROM measurements""",
            (name,),
        )
        self.connection.commit()

    def scalar_names(self) -> list[str]:
        return [row[0] for row in self.connection.execute("SELECT DISTINCT name FROM scalars ORDER BY name")]

    def scalar_rows(self, names: Sequence[str], where: str = "", parameters: Sequence = ()) -> list[dict]:
        columns = ["m.measurement_id", "m.time_collected"]
        joins = []
        for index, name in enumerate(names):
            alias = f"s{index}"
            joins.append(f"LEFT JOIN scalars {alias} ON {alias}.measurement_id = m.measurement_id AND {alias}.name = ?")
            columns.append(f"{alias}.value AS value_{index}")
        query = f"SELECT {', '.join(columns)} FROM measurements m {' '.join(joins)} {where} ORDER BY m.measurement_id"
        rows = self.connection.execute(query, tuple(names) + tuple(parameters)).fetchall()
        return [dict(zip(["measurement_id", "time_collected"] + [f"value_{i}" for i in range(len(names))], row)) for row in rows]

    def arrays(self, measurement_id: int, names: Sequence[str]) -> dict[str, np.ndarray]:
        placeholders = ",".join("?" for _ in names)
        rows = self.connection.execute(
            f"SELECT name, value FROM arrays WHERE measurement_id = ? AND name IN ({placeholders})",
            (measurement_id, *names),
        ).fetchall()
        return {name: _unpack(value) for name, value in rows}


def load_and_merge_to_sqlite(data_file_path: str, monitor_file_path_list: Sequence[str], settings_file_path_list: Sequence[str], database_path: str, keep_every: int = 1, overwrite: bool = False) -> SQLiteGradientStore:
    """Load waveform/controller data into SQLite without constructing a DataFrame."""
    if os.path.exists(database_path) and not overwrite:
        return SQLiteGradientStore(database_path)
    if os.path.exists(database_path):
        os.remove(database_path)

    data = pj.loadPickle(data_file_path)
    monitor = _read_controller_csv(monitor_file_path_list)
    settings = _read_controller_csv(settings_file_path_list)
    monitor_times = [row["Time"] for row in monitor]
    settings_times = [row["Time"] for row in settings]
    store = SQLiteGradientStore(database_path)
    waveform_keys = sorted(key for key in data[0] if str(key).startswith("voltage_"))
    time = np.asarray(data[0]["time"])
    frequency = np.fft.fftshift(np.fft.fftfreq(len(time), d=time[1] - time[0]))
    store.set_metadata("time", time)
    store.set_metadata("frequency", frequency)

    records = ((key, value) for key, value in data.items() if isinstance(key, int))
    for output_id, (measurement_id, record) in enumerate(records):
        if output_id % keep_every:
            continue
        collected = float(record["time_collected"])
        controller = _nearest(monitor, monitor_times, collected)
        setting = _nearest(settings, settings_times, collected)
        scalars = {key: value for key, value in controller.items() if key not in {"Time_str", "Time"}}
        scalars.update({key: value for key, value in setting.items() if key not in {"Time_str", "Time"}})
        scalars.update({"Time": controller["Time"], "Time_str": controller["Time_str"]})
        arrays = {key: record[key] for key in waveform_keys}
        store.add_measurement(output_id, collected, scalars=scalars, arrays=arrays)
    return store


def preprocess_sqlite(store: SQLiteGradientStore, correct_gain: bool = True, butterworth_filter: bool = True, n1_1_scaling: bool = False, bkg_subtraction: bool = False) -> SQLiteGradientStore:
    """Preprocess waveform BLOBs in SQLite, one measurement at a time."""
    waveform_keys = [name for name in store.scalar_names() if name.startswith("voltage_")]
    array_keys = [row[0] for row in store.connection.execute("SELECT DISTINCT name FROM arrays WHERE name LIKE 'voltage_%'")]
    offsets = ["voltageOffsetForward", "voltageOffsetForward", "voltageOffsetReverse", "voltageOffsetReverse"]
    sos = butter(5, 1000000, btype="highpass", analog=False, fs=500000000, output="sos")
    ids = [row[0] for row in store.connection.execute("SELECT measurement_id FROM measurements ORDER BY measurement_id")]
    for position, key in enumerate(array_keys):
        offset_key = offsets[position] if position < len(offsets) else None
        for measurement_id in ids:
            waveform = store.arrays(measurement_id, [key])[key]
            if correct_gain and offset_key:
                offset = store.scalar_rows([offset_key], "WHERE m.measurement_id = ?", (measurement_id,))[0]["value_0"]
                waveform = pj.correctVoltageByGain(waveform, offset / 10)
            if butterworth_filter:
                waveform = sosfiltfilt(sos, waveform)
            if n1_1_scaling:
                waveform = waveform / np.max(np.abs(waveform))
            if bkg_subtraction:
                waveform = waveform - np.mean(waveform[:50])
            store.set_array(measurement_id, key, waveform, commit=False)
        store.connection.commit()
    store.set_metadata("preprocessed", True)
    return store


def calculate_mean_diff(store: SQLiteGradientStore, overwrite: bool = False) -> SQLiteGradientStore:
    """Calculate temperature mean and difference using a SQL UPDATE."""
    if not overwrite and store.get_metadata("mean_diff_calculations", False):
        return store
    store.set_scalar_sql("mean_T (C)", "(SELECT AVG(value) FROM scalars WHERE measurement_id = m.measurement_id AND name IN ('1000.1: CH1 Object', '1000.2: CH2 Object'))")
    store.set_scalar_sql("diff_T (C)", "(SELECT value FROM scalars WHERE measurement_id = m.measurement_id AND name = '1000.1: CH1 Object') - (SELECT value FROM scalars WHERE measurement_id = m.measurement_id AND name = '1000.2: CH2 Object')")
    store.set_metadata("mean_diff_calculations", True)
    return store


def calculate_waveform_features(store: SQLiteGradientStore, overwrite: bool = False) -> SQLiteGradientStore:
    """Calculate waveform features and persist every result in SQLite."""
    if not overwrite and store.get_metadata("waveform_features_calculated", False):
        return store
    time = store.get_metadata("time")
    waveform_keys = [row[0] for row in store.connection.execute("SELECT DISTINCT name FROM arrays WHERE name LIKE 'voltage_%'")]
    ids = [row[0] for row in store.connection.execute("SELECT measurement_id FROM measurements ORDER BY measurement_id")]
    for key in waveform_keys:
        for measurement_id in ids:
            waveform = store.arrays(measurement_id, [key])[key]
            features = {
                "amplitude_" + key + " (mV)": pj.maxMinusMin(waveform),
                "Hilbert_ToF_" + key + " (ns)": pj.envelopeThresholdTOF(waveform, time, 0.15),
                "Hilbert_noise_ToF_" + key + " (ns)": pj.firstIndexAboveNoise(waveform),
                "max_" + key + " (mV)": float(np.max(np.abs(waveform))),
            }
            for name, value in features.items():
                store.set_scalar(measurement_id, name, value, commit=False)
            store.set_array(measurement_id, "Hilbert_window_" + key, pj.hilbertEnvelope(waveform), commit=False)
        store.connection.commit()
    store.set_metadata("waveform_features_calculated", True)
    return store


def calculate_fft_magnitude_features(store: SQLiteGradientStore, overwrite: bool = False) -> SQLiteGradientStore:
    """Calculate FFT and magnitude arrays from SQLite waveform BLOBs."""
    if not overwrite and store.get_metadata("fft_magnitude_features_calculated", False):
        return store
    f_len = len(store.get_metadata("time"))
    waveform_keys = [row[0] for row in store.connection.execute("SELECT DISTINCT name FROM arrays WHERE name LIKE 'voltage_%'")]
    ids = [row[0] for row in store.connection.execute("SELECT measurement_id FROM measurements ORDER BY measurement_id")]
    for key in waveform_keys:
        for measurement_id in ids:
            fft_value = np.fft.fftshift(np.fft.fft(store.arrays(measurement_id, [key])[key])) / f_len
            store.set_array(measurement_id, "fft_" + key, fft_value, commit=False)
            store.set_array(measurement_id, "fft_magnitude_" + key, np.abs(fft_value), commit=False)
        store.connection.commit()
    store.set_metadata("fft_magnitude_features_calculated", True)
    return store


def calculate_phase_group_delay_features(store: SQLiteGradientStore, overwrite: bool = False) -> SQLiteGradientStore:
    """Calculate phase and group delay arrays from SQLite FFT BLOBs."""
    if not overwrite and store.get_metadata("phase_group_delay_features_calculated", False):
        return store
    frequency = store.get_metadata("frequency")
    fft_keys = [row[0] for row in store.connection.execute("SELECT DISTINCT name FROM arrays WHERE name LIKE 'fft_voltage_%'")]
    ids = [row[0] for row in store.connection.execute("SELECT measurement_id FROM measurements ORDER BY measurement_id")]
    for key in fft_keys:
        for measurement_id in ids:
            phase = np.unwrap(np.angle(store.arrays(measurement_id, [key])[key]))
            delay = -np.gradient(phase, frequency) / (2 * np.pi)
            store.set_array(measurement_id, key.replace("fft_", "fft_phase_"), phase, commit=False)
            store.set_array(measurement_id, key.replace("fft_", "fft_group_delay_"), delay, commit=False)
        store.connection.commit()
    store.set_metadata("phase_group_delay_features_calculated", True)
    return store


def filter_measurements(store: SQLiteGradientStore, tolerance: float = 0.1) -> list[int]:
    """Return measurement ids passing the controller target tolerance in SQL."""
    query = """
        SELECT m.measurement_id
        FROM measurements AS m
        JOIN scalars AS o1 ON o1.measurement_id = m.measurement_id AND o1.name = '1000.1: CH1 Object'
        JOIN scalars AS o2 ON o2.measurement_id = m.measurement_id AND o2.name = '1000.2: CH2 Object'
        JOIN scalars AS t1 ON t1.measurement_id = m.measurement_id AND t1.name = '3000.1: CH1 Target'
        JOIN scalars AS t2 ON t2.measurement_id = m.measurement_id AND t2.name = '3000.2: CH2 Target'
        WHERE ABS(o1.value - t1.value) < ? AND ABS(o2.value - t2.value) < ?
        ORDER BY m.measurement_id
    """
    return [row[0] for row in store.connection.execute(query, (tolerance, tolerance))]


def scalar_matrix(store: SQLiteGradientStore, names: Sequence[str], measurement_ids: Sequence[int] | None = None) -> np.ndarray:
    """Export selected SQLite scalar columns as a NumPy matrix for sklearn."""
    if measurement_ids is None:
        rows = store.scalar_rows(names)
    else:
        placeholders = ",".join("?" for _ in measurement_ids)
        rows = store.scalar_rows(names, f"WHERE m.measurement_id IN ({placeholders})", measurement_ids)
    return np.asarray([[row[f"value_{i}"] for i in range(len(names))] for row in rows], dtype=float)


def calculate_avgs_by_T(store: SQLiteGradientStore, temps: Iterable[tuple[float, float]], keys: Sequence[str], crop: tuple[int, int] | None = None, polyfit_degree: int = 1) -> dict:
    """Calculate temperature-group means, standard deviations, and fits with SQL selection."""
    result = {key: {"avg": [], "err": [], "params": None, "fits": [], "residuals": []} for key in keys}
    dT = []
    T_avg = []
    for target_forward, target_reverse in temps:
        dT.append(abs(target_forward - target_reverse))
        T_avg.append((target_forward + target_reverse) / 2)
        where = """
            WHERE EXISTS (SELECT 1 FROM scalars WHERE measurement_id = m.measurement_id AND name = '3000.1: CH1 Target' AND value = ?)
              AND EXISTS (SELECT 1 FROM scalars WHERE measurement_id = m.measurement_id AND name = '3000.2: CH2 Target' AND value = ?)
              AND EXISTS (SELECT 1 FROM scalars WHERE measurement_id = m.measurement_id AND name = '2010.1: Output Enable' AND text_value = 'ON')
              AND EXISTS (SELECT 1 FROM scalars WHERE measurement_id = m.measurement_id AND name = '2010.2: Output Enable' AND text_value = 'ON')
        """
        params = [target_forward, target_reverse, target_forward, target_reverse]
        if crop:
            where += " AND m.measurement_id >= ? AND m.measurement_id < ?"
            params.extend(crop)
        for key in keys:
            row = store.connection.execute(
                f"SELECT AVG(s.value), COUNT(s.value), AVG(s.value * s.value) - AVG(s.value) * AVG(s.value) FROM measurements m JOIN scalars s ON s.measurement_id = m.measurement_id AND s.name = ? {where}",
                (key, *params),
            ).fetchone()
            if row[1]:
                result[key]["avg"].append(row[0])
                result[key]["err"].append(float(np.sqrt(max(row[2] or 0, 0))))
            else:
                result[key]["avg"].append(np.nan)
                result[key]["err"].append(np.nan)
    valid = np.isfinite(dT)
    x = np.asarray(dT)[valid]
    for key in keys:
        y = np.asarray(result[key]["avg"], dtype=float)
        keep = np.isfinite(y)
        result[key]["params"] = np.polyfit(x[keep], y[keep], polyfit_degree) if keep.sum() > polyfit_degree else np.array([])
        result[key]["fits"] = np.polyval(result[key]["params"], x) if result[key]["params"].size else []
        result[key]["residuals"] = y[keep] - result[key]["fits"] if result[key]["params"].size else []
    result["dT"] = dT
    result["T_avg"] = T_avg
    return result
