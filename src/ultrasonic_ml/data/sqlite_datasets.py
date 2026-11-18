from typing import Any   
from pathlib import Path
import io
import sqlite3
import numpy as np

class AcousticSQlite: 
    """Base class for SQLite acoustics data and analysis."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the AcousticSQlite.

        Parameters
        ----------
        path : str | Path
            The path to the SQLite database file.
        """
        
        print("Establishing SQLite connection to database...")
        self.types_dict = {
            "REAL": float,
            "TEXT": str,
            "INTEGER": int,
            "BLOB": bytes,
            "NULL": type(None),
        }
        self.path = path
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self.acoustics_table = "acoustics"
        self.parameters_table = "parameters"

        self.connect()
        # print('setting parameters')
        self.waveform_columns = self.get_waveform_columns()
        self.parameters = self.get_sqlite_parameters()
        self.initialize_frequency_parameters()

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    def __enter__(self) -> AcousticSQlite:
        """Enter the context manager.

        Returns
        -------
        AcousticSQlite
            The AcousticSQlite instance.
        """
        
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager and close the database connection.

        Parameters
        ----------
        *args : Any
            The exception type, value, and traceback.
        """
        self.close()

    def __len__(self) -> int:
        """Get the number of acquisitions in the acoustics table.

        Returns
        -------
        int
            The number of acquisitions.
        """
        return self.get_acquisition_count()

    def print_schema(self) -> None:
        """Pretty-print the database's table and column structure.
        """
        tables = self.get_tables()
        for i, table in enumerate(tables):
            is_last_table = i == len(tables) - 1
            print(f"{'└── ' if is_last_table else '├── '}{table}")

            columns = self.get_columns(table)
            prefix = "    " if is_last_table else "│   "
            for j, column in enumerate(columns):
                is_last_column = j == len(columns) - 1
                print(f"{prefix}{'└── ' if is_last_column else '├── '}{column}")

    def connect(self) -> None:
        """Connect to the SQLite database.
        """
        self.connection = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

    def close(self) -> None:
        """Close the SQLite database connection.
        """
        if self.connection:
            self.connection.close()
        self.connection = self.cursor = None

    def get_tables(self) -> list[str]:
        """Get the list of tables in the SQLite database.

        Returns
        -------
        list[str]
            A list of table names.
        """
        rows = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [r[0] for r in rows]

    def get_columns(self, table: str) -> list[str]:
        """Get the list of columns in a table.

        Parameters
        ----------
        table : str
            The name of the table.

        Returns
        -------
        list[str]
            A list of column names.
        """
        rows = self.connection.execute(f"PRAGMA table_info({self._quote(table)})").fetchall()
        return [r[1] for r in rows]

    def fetch_column(self, column: str, table: str = "acoustics") -> list[Any]:
        """Fetch all values from a specific column in a table.

        Parameters
        ----------
        column : str
            The name of the column.
        table : str, optional
            The name of the table, by default "acoustics".

        Returns
        -------
        list[Any]
            A list of values from the specified column.
        """
        self._check_column(table, column)
        query = f"SELECT {self._quote(column)} FROM {self._quote(table)}"
        return [r[0] for r in self.connection.execute(query)]

    def fetch_value(self, column: str, row: int, table: str = "acoustics") -> Any:
        """Fetch a single value from a specific column and row in a table.

        Parameters
        ----------
        column : str
            The name of the column.
        row : int
            The row index.
        table : str, optional
            The name of the table, by default "acoustics".

        Returns
        -------
        Any
            The value from the specified column and row.
        """
        self._check_column(table, column)
        query = f"SELECT {self._quote(column)} FROM {self._quote(table)} LIMIT 1 OFFSET ?"
        result = self.connection.execute(query, (row,)).fetchone()
        if result is None:
            raise IndexError(f"Row {row} does not exist.")
        return result[0]

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def get_sqlite_parameters(self) -> dict[str, Any]:
        """Get the parameters from the parameters table.

        Returns
        -------
        dict[str, Any]
            A dictionary of parameters.
        """
        columns = self.get_columns(self.parameters_table)
        if not columns:
            return {}
        query = f"SELECT {','.join(map(self._quote, columns))} FROM {self._quote(self.parameters_table)} LIMIT 1"
        row = self.connection.execute(query).fetchone()
        return dict(zip(columns, row)) if row else {}

    def initialize_frequency_parameters(self) -> None:
        """Initialize frequency-related parameters from the database.
        """
        self.f = self.parameters.get("transducerFrequency")
        self.f = float(self.f) * 1e6 if self.f is not None else None

        try:
            time = self.fetch_time()
            self.dt = float(np.median(np.diff(time))) * 1e-9
            self.f_s = 1 / self.dt
        except (KeyError, ValueError, IndexError):
            self.dt = self.f_s = None

        self.omega_scaled = self.f * self.dt if self.f is not None and self.dt is not None else None

        values = {"omega_scaled": self.omega_scaled, "dt": self.dt, "f_s": self.f_s}
        for key, value in values.items():
            if value is not None:
                self._add_column(key, self.parameters_table)
                self._write_parameter(key, value)

        self.parameters.update({k: v for k, v in values.items() if v is not None})

    # -------------------------------------------------------------------------
    # Raw Waveforms
    # -------------------------------------------------------------------------

    def store_acoustics_values(self, collection_index: int, column: str, value: Any) -> None:
        """Store a single value in the acoustics table.

        Parameters
        ----------
        collection_index : int
            The index of the collection.
        column : str
            The name of the column to store the value in.
        value : Any
            The value to store.
        """
        self.store_acoustics_values_batch([collection_index], column, [value])

    def store_acoustics_values_batch(self, collection_index: int, column: str, value: Any) -> None:
        """Store values in the analysis table.

        Parameters
        ----------
        collection_index : int
            The index of the collection.
        """
        if isinstance(value, np.ndarray):
            value = self.serialize_array(value)

        query = f"""
            INSERT INTO {self._quote(self.acoustics_table)}
            (collection_index,{self._quote(column)})
            VALUES (?,?)
            ON CONFLICT(collection_index) DO UPDATE SET
                {self._quote(column)} = excluded.{self._quote(column)}
        """
        self.connection.executemany(query, zip(collection_index, value))
        self.connection.commit()

    def fetch_waveform(self, waveform: str, row: int) -> np.ndarray:
        """Fetch a waveform from the acoustics table.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.
        row : int
            The row index of the acquisition.

        Returns
        -------
        np.ndarray
            The waveform as a NumPy array.
        """
        
        self._check_column('acoustics', waveform)
        return self.deserialize_array(self.fetch_value(waveform, row))

    def fetch_waveform_batch(self, waveform: str, rows: Any) -> list[np.ndarray]:
        """Fetch a batch of waveforms from the acoustics table.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.
        rows : Any
            An iterable of row indices.

        Returns
        -------
        list[np.ndarray]
            A list of waveforms as NumPy arrays.
        """
        return [self.fetch_waveform(waveform, int(row)) for row in rows]

    def get_waveform_columns(self) -> list[str]:
        """Get the list of waveform columns in the acoustics table.

        Returns
        -------
        list[str]
            A list of waveform column names.
        """
        excluded = {"collection_index", "X", "Z", "time", "time_collected", "frequency"}
        return [c for c in self.get_columns(self.acoustics_table) if c.startswith("voltage") and c not in excluded]

    def fetch_time(self, row: int = 0) -> np.ndarray:
        """Fetch the time array from the acoustics table.

        Parameters
        ----------
        row : int, optional
            The row index of the acquisition, by default 0.

        Returns
        -------
        np.ndarray
            The time array as a NumPy array.
        """
        return self.deserialize_array(self.fetch_value("time", row))

    def get_acquisition_count(self) -> int:
        """Get the total number of acquisitions in the acoustics table.

        Returns
        -------
        int
            The total number of acquisitions.
        """
        
        query = f"SELECT COUNT(*) FROM {self._quote(self.acoustics_table)}"
        return int(self.connection.execute(query).fetchone()[0])

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    @staticmethod
    def serialize_array(array: np.ndarray) -> bytes:
        """Serialize a NumPy array to bytes for storage in the SQLite database.

        Parameters
        ----------
        array : np.ndarray
            The NumPy array to serialize.

        Returns
        -------
        bytes
            The serialized byte representation of the array.
        """
            
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def deserialize_array(blob: Any) -> np.ndarray:
        """Deserialize bytes from the SQLite database back into a NumPy array.

        Parameters
        ----------
        blob : Any
            The byte representation of the array.

        Returns
        -------
        np.ndarray
            The deserialized NumPy array.
        """
            
        if isinstance(blob, np.ndarray):
            return blob
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        if isinstance(blob, bytearray):
            blob = bytes(blob)
        if not isinstance(blob, bytes):
            return np.asarray(blob)

        try:
            return np.load(io.BytesIO(blob), allow_pickle=False)
        except (ValueError, EOFError):
            return np.frombuffer(blob, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _delete_column(self, column: str, table: str) -> None:
        """Delete a column from a table in the SQLite database.

        Parameters
        ----------
        column : str
            The name of the column to delete.
        table : str
            The name of the table from which to delete the column.
        """
            
        if column in self.get_columns(table):
            self.connection.execute(f"ALTER TABLE {self._quote(table)} DROP COLUMN {self._quote(column)}")

    def _add_column(self, column: str, table: str, type: str='REAL') -> None:
        """Add a new column to the parameters table if it does not already exist.

        Parameters
        ----------
        column : str
            The name of the column to add.
        """
            
        if column not in self.get_columns(table):
            self.connection.execute(f"ALTER TABLE {self._quote(table)} ADD COLUMN {self._quote(column)} {type}")

    def _write_parameter(self, column: str, value: Any) -> None:
        """Write a parameter value to the parameters table.

        Parameters
        ----------
        column : str
            The name of the parameter column.
        value : Any
            The value to write to the parameter column.
        """
            
        self.connection.execute(f"UPDATE {self._quote(self.parameters_table)} SET {self._quote(column)}=?", (value,))
        self.connection.commit()

    def _check_column(self, table: str, column: str) -> None:
        """Check if a column exists in a table.

        Parameters
        ----------
        table : str
            The name of the table.
        column : str
            The name of the column.

        Raises
        ------
        ValueError
            If the column does not exist in the table.
        """
            
        if column not in self.get_columns(table): raise ValueError(f"Column '{column}' does not exist in '{table}'")

    @staticmethod
    def _quote(identifier: str) -> str:
        """Replace quotes with double quotes in an identifier (table or column name) for use in SQL queries.

        Parameters
        ----------
        identifier : str
            The identifier to quote.

        Returns
        -------
        str
            The quoted identifier.
        """
            
        return '"' + identifier.replace('"', '""') + '"'

