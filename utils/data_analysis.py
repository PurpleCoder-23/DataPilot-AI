from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json"}
MAX_FILE_SIZE_MB = 50


class DataAnalysisError(Exception):
    """Raised when a dataset cannot be loaded or processed."""


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

def load_dataset(source: Any) -> pd.DataFrame:
    """Load CSV, Excel, or JSON data into a DataFrame.

    source may be a file path, Streamlit UploadedFile,
    or file-like object.
    """
    try:
        filename = getattr(source, "name", None) or str(source)
        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_FORMATS:
            raise DataAnalysisError(
                f"Unsupported file format: {extension or '(no extension)'}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        file_size = getattr(source, "size", None)

        if file_size is None and isinstance(source, (str, Path)):
            file_size = Path(source).stat().st_size

        if (
            file_size is not None
            and file_size > MAX_FILE_SIZE_MB * 1024 * 1024
        ):
            raise DataAnalysisError(
                f"File exceeds the {MAX_FILE_SIZE_MB} MB limit."
            )

        if hasattr(source, "seek"):
            source.seek(0)

        if extension == ".csv":
            df = pd.read_csv(source)

        elif extension in {".xlsx", ".xls"}:
            df = pd.read_excel(source)

        else:
            df = pd.read_json(source)

        if df.empty:
            raise DataAnalysisError(
                "The dataset contains no rows."
            )

        if len(df.columns) == 0:
            raise DataAnalysisError(
                "The dataset contains no columns."
            )

        return df

    except DataAnalysisError:
        raise

    except Exception as exc:
        raise DataAnalysisError(
            f"Unable to load dataset: {exc}"
        ) from exc


# ---------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------

def preview_dataset(
    df: pd.DataFrame,
    rows: int = 10
) -> dict:
    """Return a preview and basic schema information."""

    if rows < 1:
        raise ValueError(
            "Preview rows must be greater than zero."
        )

    return {
        "preview": df.head(rows),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return basic dataset quality and memory information."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum()
            / (1024 * 1024),
            2
        ),
    }


# ---------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a compact profile of dataset quality
    and column characteristics.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame"
        )

    missing_count = df.isna().sum()

    profile = pd.DataFrame({
        "column": df.columns.astype(str),
        "data_type": df.dtypes.astype(str).values,
        "missing_count": missing_count.values,
        "missing_percent": (
            missing_count / max(len(df), 1) * 100
        ).round(2).values,
        "unique_values": df.nunique(
            dropna=True
        ).values,
    })

    numeric_columns = (
        df.select_dtypes(include="number")
        .columns.tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
                "string"
            ]
        )
        .columns.tolist()
    )

    return {
        "overview": get_dataset_info(df),
        "columns": profile,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


# ---------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------

def numerical_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().T


# ---------------------------------------------------------
# Categorical summary
# ---------------------------------------------------------

def categorical_summary(
    df: pd.DataFrame,
    top_n: int = 10
) -> dict[str, pd.DataFrame]:
    """Return frequency tables for categorical columns."""

    if top_n < 1:
        raise ValueError(
            "top_n must be greater than zero"
        )

    categorical_df = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
            "string"
        ]
    )

    summaries = {}

    for column in categorical_df.columns:

        counts = (
            categorical_df[column]
            .value_counts(dropna=False)
            .head(top_n)
            .rename_axis("value")
            .reset_index(name="count")
        )

        summaries[str(column)] = counts

    return summaries


# ---------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------

def correlation_matrix(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Return correlations between numeric columns."""

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr()


# ---------------------------------------------------------
# Outlier summary
# ---------------------------------------------------------

def outlier_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Detect potential numeric outliers using 1.5×IQR.

    This function reports outliers but does not remove
    or modify the original dataset.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    results = []

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if series.empty:

            results.append({
                "column": str(column),
                "outlier_count": 0,
                "outlier_percent": 0.0,
                "lower_bound": None,
                "upper_bound": None,
            })

            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            (
                (series < lower_bound)
                | (series > upper_bound)
            ).sum()
        )

        results.append({
            "column": str(column),
            "outlier_count": outlier_count,
            "outlier_percent": round(
                outlier_count / len(series) * 100,
                2
            ),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        })

    return pd.DataFrame(results)


# =========================================================
# DATA CLEANING FUNCTIONS
# =========================================================


# ---------------------------------------------------------
# Remove duplicate rows
# ---------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    keep: str = "first"
) -> pd.DataFrame:
    """Return a copy with duplicate rows removed.

    keep:
        'first' - keep first occurrence
        'last'  - keep last occurrence
        False   - remove all duplicate occurrences
    """

    if keep not in {"first", "last", False}:
        raise ValueError(
            "keep must be 'first', 'last', or False"
        )

    return (
        df.copy()
        .drop_duplicates(keep=keep)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Handle missing values
# ---------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop_rows",
    fill_value: Any = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Handle missing values in selected columns.

    Supported strategies:
        drop_rows
        drop_columns
        fill_value
        mean
        median
        mode
    """

    result = df.copy()

    selected = (
        columns
        if columns is not None
        else result.columns.tolist()
    )

    missing_columns = [
        col for col in selected
        if col not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unknown column(s): {missing_columns}"
        )

    # Drop rows containing missing values
    if strategy == "drop_rows":

        return (
            result.dropna(subset=selected)
            .reset_index(drop=True)
        )

    # Drop selected columns that contain missing values
    if strategy == "drop_columns":

        columns_with_missing = [
            col for col in selected
            if result[col].isna().any()
        ]

        return result.drop(
            columns=columns_with_missing
        )

    # Fill missing values with a custom value
    if strategy == "fill_value":

        if fill_value is None:
            raise ValueError(
                "fill_value must be provided "
                "for this strategy"
            )

        result[selected] = (
            result[selected].fillna(fill_value)
        )

        return result

    # Fill numeric columns using mean or median
    if strategy in {"mean", "median"}:

        for col in selected:

            if not pd.api.types.is_numeric_dtype(
                result[col]
            ):
                raise ValueError(
                    f"Column '{col}' is not numeric; "
                    f"{strategy} is not applicable"
                )

            if strategy == "mean":
                value = result[col].mean()

            else:
                value = result[col].median()

            result[col] = result[col].fillna(value)

        return result

    # Fill missing values using mode
    if strategy == "mode":

        for col in selected:

            modes = result[col].mode(
                dropna=True
            )

            if not modes.empty:

                result[col] = result[col].fillna(
                    modes.iloc[0]
                )

        return result

    raise ValueError(
        "Unsupported strategy. Choose "
        "drop_rows, drop_columns, fill_value, "
        "mean, median, or mode."
    )


# ---------------------------------------------------------
# Drop selected columns
# ---------------------------------------------------------

def drop_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """Return a copy with selected columns removed."""

    missing_columns = [
        col for col in columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unknown column(s): {missing_columns}"
        )

    return df.copy().drop(
        columns=columns
    )


# ---------------------------------------------------------
# Rename columns
# ---------------------------------------------------------

def rename_columns(
    df: pd.DataFrame,
    mapping: Mapping[str, str]
) -> pd.DataFrame:
    """Return a copy with columns renamed.

    Example:
        {"old_name": "new_name"}
    """

    missing_columns = [
        col for col in mapping
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unknown column(s): {missing_columns}"
        )

    new_names = [
        mapping.get(col, col)
        for col in df.columns
    ]

    if len(new_names) != len(set(new_names)):
        raise ValueError(
            "Renaming would create duplicate column names"
        )

    return (
        df.copy()
        .rename(columns=dict(mapping))
    )


# ---------------------------------------------------------
# Convert column data types
# ---------------------------------------------------------

def convert_column_types(
    df: pd.DataFrame,
    dtype_mapping: Mapping[str, Any]
) -> pd.DataFrame:
    """Return a copy with selected columns converted.

    Example:
        {
            "age": "Int64",
            "price": "float",
            "date": "datetime64[ns]"
        }
    """

    missing_columns = [
        col for col in dtype_mapping
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unknown column(s): {missing_columns}"
        )

    result = df.copy()

    try:

        for column, dtype in dtype_mapping.items():

            if str(dtype).lower() in {
                "datetime",
                "datetime64",
                "datetime64[ns]"
            }:

                result[column] = pd.to_datetime(
                    result[column],
                    errors="raise"
                )

            else:

                result[column] = (
                    result[column].astype(dtype)
                )

    except (TypeError, ValueError) as exc:

        raise ValueError(
            f"Unable to convert column type: {exc}"
        ) from exc

    return result


# ---------------------------------------------------------
# Select, filter, sort, and limit dataset rows
# ---------------------------------------------------------

def select_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return a copy containing only the requested columns."""

    if not columns:
        raise ValueError("At least one column must be selected.")

    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Unknown column(s): {missing_columns}")

    return df.loc[:, columns].copy()


def filter_rows(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any = None,
) -> pd.DataFrame:
    """Filter rows using a validated comparison operator.

    Supported operators:
        equals, not_equals, contains, startswith, endswith,
        greater_than, greater_equal, less_than, less_equal,
        is_null, not_null
    """

    if column not in df.columns:
        raise ValueError(f"Unknown column: '{column}'")

    series = df[column]
    op = operator.strip().lower()

    if op == "equals":
        mask = series == value
    elif op == "not_equals":
        mask = series != value
    elif op == "contains":
        mask = series.astype("string").str.contains(
            str(value), case=False, na=False, regex=False
        )
    elif op == "startswith":
        mask = series.astype("string").str.startswith(
            str(value), na=False
        )
    elif op == "endswith":
        mask = series.astype("string").str.endswith(
            str(value), na=False
        )
    elif op == "greater_than":
        mask = series > value
    elif op == "greater_equal":
        mask = series >= value
    elif op == "less_than":
        mask = series < value
    elif op == "less_equal":
        mask = series <= value
    elif op == "is_null":
        mask = series.isna()
    elif op == "not_null":
        mask = series.notna()
    else:
        raise ValueError(
            "Unsupported filter operator. Use equals, not_equals, "
            "contains, startswith, endswith, greater_than, "
            "greater_equal, less_than, less_equal, is_null, or not_null."
        )

    return df.loc[mask].copy().reset_index(drop=True)


def sort_dataset(
    df: pd.DataFrame,
    columns: str | list[str],
    ascending: bool | list[bool] = True,
) -> pd.DataFrame:
    """Return a copy sorted by one or more existing columns."""

    sort_columns = [columns] if isinstance(columns, str) else list(columns)

    if not sort_columns:
        raise ValueError("At least one sort column must be provided.")

    missing_columns = [
        col for col in sort_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Unknown column(s): {missing_columns}")

    return (
        df.copy()
        .sort_values(
            by=sort_columns,
            ascending=ascending,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def limit_rows(
    df: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Return the first n rows without modifying the input dataset."""

    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer.")

    return df.head(n).copy().reset_index(drop=True)
