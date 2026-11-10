"""Direct SQLite processing for ultrasonic thermal-gradient experiments.

This module reads the original ``acoustics`` SQLite table directly and writes
processed waveforms and features to SQLite. It deliberately does not import
pandas, pickle, or pickleJar.
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


OUTPUT_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id INTEGER PRIMARY KEY,
    source_collection_index INTEGER UNIQUE,
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
CREATE INDEX IF NOT EXISTS idx_scalar_name_value ON scalars(name, value);
CREATE INDEX IF NOT EXISTS idx_measurement_time ON measurements(time_collected);
"""


def encode_array(value: np.ndarray) -> sqlite3.Binary:
    """Encode an ndarray as a SQLite-safe NumPy binary BLOB."""
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return sqlite3.Binary(stream.getvalue())


def decode_array(value: bytes) -> np.ndarray:
    """Decode an array BLOB written by :func:`encode_array`."""
    return np.load(io.BytesIO(value), allow_pickle=False)


def sqlite_array_converter(value: bytes) -> np.ndarray:
    """Compatibility converter for databases declaring columns as ``array``."""
    return decode_array(value)


def open_source(database_path: str | os.PathLike) -> sqlite3.Connection:
    """Open a source experiment database with NumPy array conversion enabled."""
    sqlite3.register_converter("array", decode_array)
    connection = sqlite3.connect(str(database_path), detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row
    connection.text_factory = str
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'acoustics'").fetchone()
    if connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'acoustics'").fetchone() is None:
        connection.close()
        raise ValueError(f"{database_path} does not contain an acoustics table")
    return connection


def open_output(database_path: str | os.PathLike, overwrite: bool = False) -> sqlite3.Connection:
    """Create or open a processed SQLite database."""
    database_path = str(database_path)
    if overwrite and os.path.exists(database_path):
        os.remove(database_path)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(OUTPUT_SCHEMA)
    connection.commit()
    return connection


def _array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if value is None:
        raise ValueError("Expected a waveform array, received NULL")
    return decode_array(value)


def _scalar(value):
    if value is None or value == "":
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, str(value)


def _correct_voltage_by_gain(data: np.ndarray, gain: float) -> np.ndarray:
    signs = np.sign(data)
    log_data = np.log10(np.abs(data) + 10e-10)
    return signs * 10 ** (log_data - gain / 20)


def _hilbert_envelope(data: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(data))


def _first_index_above_noise(data: np.ndarray, standard_deviations: float = 25) -> int:
    noise = data[: len(data) // 20]
    threshold = np.mean(noise) + standard_deviations * np.std(noise)
    return int(np.argmax(data > threshold))


def _envelope_tof(data: np.ndarray, time: np.ndarray, threshold: float = 0.15) -> float:
    envelope = _hilbert_envelope(data)
    threshold_value = threshold * np.max(envelope)
    return float(time[np.argmax(envelope > threshold_value)])


def _set_metadata(connection: sqlite3.Connection, key: str, value) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, encode_array(value) if isinstance(value, np.ndarray) else sqlite3.Binary(str(value).encode())),
    )


def _controller_rows(paths: Sequence[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream, delimiter=";"):
                if not row.get("Time"):
                    continue
                original_time = row["Time"]
                row["Time"] = datetime.strptime(original_time, "%m/%d/%Y %I:%M:%S %p").timestamp()
                row["Time"] += float(row.get("Milliseconds") or 0) / 1000
                row["Time_str"] = original_time
                row.pop("Milliseconds", None)
                rows.append(row)
    return sorted(rows, key=lambda row: row["Time"])


def _nearest(rows: Sequence[dict], timestamps: Sequence[float], value: float) -> dict:
    position = bisect_left(timestamps, value)
    if position == 0:
        return rows[0]
    if position == len(rows):
        return rows[-1]
    before, after = rows[position - 1], rows[position]
    return before if value - before["Time"] <= after["Time"] - value else after


def _insert_value(connection, measurement_id: int, name: str, value, commit: bool = False) -> None:
    numeric, text = _scalar(value)
    connection.execute(
        "INSERT OR REPLACE INTO scalars(measurement_id, name, value, text_value) VALUES (?, ?, ?, ?)",
        (measurement_id, name, numeric, text),
    )
    if commit:
        connection.commit()


def _insert_array(connection, measurement_id: int, name: str, value: np.ndarray, commit: bool = False) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO arrays(measurement_id, name, value) VALUES (?, ?, ?)",
        (measurement_id, name, encode_array(value)),
    )
    if commit:
        connection.commit()


def write_array(connection: sqlite3.Connection, measurement_id: int, name: str, value: np.ndarray, commit: bool = False) -> None:
    """Write a derived NumPy array to the SQLite output store."""
    _insert_array(connection, measurement_id, name, value, commit=commit)


def import_sqlite(source_path: str | os.PathLike, output_path: str | os.PathLike, monitor_paths: Sequence[str] = (), settings_paths: Sequence[str] = (), keep_every: int = 1, overwrite: bool = False) -> sqlite3.Connection:
    """Import source SQLite rows directly into the processed SQLite store."""
    source = open_source(source_path)
    output = open_output(output_path, overwrite=overwrite)
    monitor = _controller_rows(monitor_paths)
    settings = _controller_rows(settings_paths)
    monitor_times = [row["Time"] for row in monitor]
    settings_times = [row["Time"] for row in settings]

    columns = [row["name"] for row in source.execute("PRAGMA table_info(acoustics)")]
    array_columns = [name for name in columns if name.startswith("voltage") or name == "time"]
    scalar_columns = [name for name in columns if name not in array_columns and name != "collection_index"]
    query = "SELECT " + ", ".join('"' + name.replace('"', '""') + '"' for name in columns) + " FROM acoustics ORDER BY collection_index"
    rows = source.execute(query)
    first_time = None

    for row_number, row in enumerate(rows):
        if row_number % keep_every:
            continue
        source_index = int(row["collection_index"])
        time_collected = float(row["time_collected"])
        measurement_id = row_number // keep_every
        output.execute(
            "INSERT OR REPLACE INTO measurements(measurement_id, source_collection_index, time_collected) VALUES (?, ?, ?)",
            (measurement_id, source_index, time_collected),
        )
        for name in scalar_columns:
            if row[name] is not None:
                _insert_value(output, measurement_id, name, row[name])
        if monitor:
            controller = _nearest(monitor, monitor_times, time_collected)
            for name, value in controller.items():
                _insert_value(output, measurement_id, name, value)
        if settings:
            controller_settings = _nearest(settings, settings_times, time_collected)
            for name, value in controller_settings.items():
                _insert_value(output, measurement_id, name, value)
        for name in array_columns:
            if row[name] is not None:
                _insert_array(output, measurement_id, name, _array(row[name]))
        if first_time is None and row["time"] is not None:
            first_time = _array(row["time"])

    if first_time is not None:
        _set_metadata(output, "time", first_time)
        _set_metadata(output, "frequency", np.fft.fftshift(np.fft.fftfreq(len(first_time), d=first_time[1] - first_time[0])))
    _set_metadata(output, "source_path", str(source_path))
    output.commit()
    source.close()
    return output


def waveform_names(connection: sqlite3.Connection) -> list[str]:
    return [row[0] for row in connection.execute("SELECT DISTINCT name FROM arrays WHERE name LIKE 'voltage%'")]


def measurement_ids(connection: sqlite3.Connection) -> list[int]:
    return [row[0] for row in connection.execute("SELECT measurement_id FROM measurements ORDER BY measurement_id")]


def read_array(connection: sqlite3.Connection, measurement_id: int, name: str) -> np.ndarray:
    row = connection.execute("SELECT value FROM arrays WHERE measurement_id = ? AND name = ?", (measurement_id, name)).fetchone()
    if row is None:
        raise KeyError(f"Array {name!r} not found for measurement {measurement_id}")
    return decode_array(row[0])


def preprocess(connection: sqlite3.Connection, correct_gain: bool = True, butterworth_filter: bool = True, n1_1_scaling: bool = False, bkg_subtraction: bool = False) -> None:
    """Preprocess each waveform directly from and back into SQLite."""
    offsets = ["voltageOffsetForward", "voltageOffsetForward", "voltageOffsetReverse", "voltageOffsetReverse"]
    filter_sos = butter(5, 1000000, btype="highpass", analog=False, fs=500000000, output="sos")
    for index, name in enumerate(waveform_names(connection)):
        offset_name = offsets[index] if index < len(offsets) else None
        for measurement_id in measurement_ids(connection):
            waveform = read_array(connection, measurement_id, name)
            if correct_gain and offset_name:
                offset = connection.execute("SELECT value FROM scalars WHERE measurement_id = ? AND name = ?", (measurement_id, offset_name)).fetchone()
                if offset is not None:
                    waveform = _correct_voltage_by_gain(waveform, float(offset[0]) / 10)
            if butterworth_filter:
                waveform = sosfiltfilt(filter_sos, waveform)
            if n1_1_scaling:
                waveform = waveform / np.max(np.abs(waveform))
            if bkg_subtraction:
                waveform = waveform - np.mean(waveform[:50])
            _insert_array(connection, measurement_id, name, waveform)
    _set_metadata(connection, "preprocessed", "1")
    connection.commit()


def calculate_mean_diff(connection: sqlite3.Connection, overwrite: bool = False) -> None:
    """Calculate temperature mean and difference with SQL expressions."""
    if not overwrite and connection.execute("SELECT 1 FROM metadata WHERE key = 'mean_diff_calculations'").fetchone():
        return
    connection.execute(
        """INSERT OR REPLACE INTO scalars(measurement_id, name, value)
        SELECT m.measurement_id, 'mean_T (C)', AVG(s.value)
        FROM measurements m JOIN scalars s ON s.measurement_id = m.measurement_id
        WHERE s.name IN ('1000.1: CH1 Object', '1000.2: CH2 Object')
        GROUP BY m.measurement_id"""
    )
    connection.execute(
        """INSERT OR REPLACE INTO scalars(measurement_id, name, value)
        SELECT m.measurement_id, 'diff_T (C)',
            (SELECT value FROM scalars WHERE measurement_id = m.measurement_id AND name = '1000.1: CH1 Object') -
            (SELECT value FROM scalars WHERE measurement_id = m.measurement_id AND name = '1000.2: CH2 Object')
        FROM measurements m"""
    )
    _set_metadata(connection, "mean_diff_calculations", "1")
    connection.commit()


def calculate_waveform_features(connection: sqlite3.Connection, overwrite: bool = False) -> None:
    """Calculate scalar waveform features and Hilbert envelopes in SQLite."""
    if not overwrite and connection.execute("SELECT 1 FROM metadata WHERE key = 'waveform_features_calculated'").fetchone():
        return
    time_row = connection.execute("SELECT value FROM metadata WHERE key = 'time'").fetchone()
    if time_row is None:
        raise ValueError("The output database has no time metadata")
    time = decode_array(time_row[0])
    for name in waveform_names(connection):
        if name == "time":
            continue
        for measurement_id in measurement_ids(connection):
            waveform = read_array(connection, measurement_id, name)
            envelope = _hilbert_envelope(waveform)
            values = {
                "amplitude_" + name + " (mV)": float(np.max(waveform) - np.min(waveform)),
                "max_" + name + " (mV)": float(np.max(np.abs(waveform))),
                "Hilbert_ToF_" + name + " (ns)": _envelope_tof(waveform, time),
                "Hilbert_noise_ToF_" + name + " (ns)": float(time[_first_index_above_noise(envelope)]),
            }
            for feature_name, value in values.items():
                _insert_value(connection, measurement_id, feature_name, value)
            _insert_array(connection, measurement_id, "Hilbert_window_" + name, envelope)
    _set_metadata(connection, "waveform_features_calculated", "1")
    connection.commit()


def calculate_fft_features(connection: sqlite3.Connection, overwrite: bool = False) -> None:
    """Calculate FFT magnitude, phase, and group delay arrays in SQLite."""
    if not overwrite and connection.execute("SELECT 1 FROM metadata WHERE key = 'fft_features_calculated'").fetchone():
        return
    frequency = decode_array(connection.execute("SELECT value FROM metadata WHERE key = 'frequency'").fetchone()[0])
    for name in waveform_names(connection):
        if name == "time":
            continue
        for measurement_id in measurement_ids(connection):
            fft_value = np.fft.fftshift(np.fft.fft(read_array(connection, measurement_id, name))) / len(frequency)
            phase = np.unwrap(np.angle(fft_value))
            _insert_array(connection, measurement_id, "fft_" + name, fft_value)
            _insert_array(connection, measurement_id, "fft_magnitude_" + name, np.abs(fft_value))
            _insert_array(connection, measurement_id, "fft_phase_" + name, phase)
            _insert_array(connection, measurement_id, "fft_group_delay_" + name, -np.gradient(phase, frequency) / (2 * np.pi))
    _set_metadata(connection, "fft_features_calculated", "1")
    connection.commit()


def filter_measurements(connection: sqlite3.Connection, tolerance: float = 0.1) -> list[int]:
    """Return measurement ids passing temperature-target tolerance using SQL."""
    query = """
    SELECT m.measurement_id FROM measurements m
    JOIN scalars o1 ON o1.measurement_id = m.measurement_id AND o1.name = '1000.1: CH1 Object'
    JOIN scalars o2 ON o2.measurement_id = m.measurement_id AND o2.name = '1000.2: CH2 Object'
    JOIN scalars t1 ON t1.measurement_id = m.measurement_id AND t1.name = '3000.1: CH1 Target'
    JOIN scalars t2 ON t2.measurement_id = m.measurement_id AND t2.name = '3000.2: CH2 Target'
    WHERE ABS(o1.value - t1.value) < ? AND ABS(o2.value - t2.value) < ?
    ORDER BY m.measurement_id
    """
    return [row[0] for row in connection.execute(query, (tolerance, tolerance))]


def scalar_matrix(connection: sqlite3.Connection, names: Sequence[str], ids: Sequence[int] | None = None) -> np.ndarray:
    """Export selected SQLite scalar values to NumPy for PCA or plotting."""
    ids = measurement_ids(connection) if ids is None else list(ids)
    result = []
    for measurement_id in ids:
        row = []
        for name in names:
            value = connection.execute("SELECT value FROM scalars WHERE measurement_id = ? AND name = ?", (measurement_id, name)).fetchone()
            row.append(np.nan if value is None else value[0])
        result.append(row)
    return np.asarray(result, dtype=float)
