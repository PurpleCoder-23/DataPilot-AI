
import pandas as pd
import pytest

from utils.data_analysis import (
    DataAnalysisError,
    load_dataset,
    preview_dataset,
    get_dataset_info,
)


def test_load_csv(tmp_path):
    file_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    }).to_csv(file_path, index=False)

    df = load_dataset(file_path)

    assert len(df) == 2
    assert list(df.columns) == ["name", "age"]


def test_preview_dataset():
    df = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    })

    result = preview_dataset(df, rows=1)

    assert result["total_rows"] == 2
    assert result["total_columns"] == 2
    assert len(result["preview"]) == 1


def test_dataset_info():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", None],
        "age": [25, 30, 35],
    })

    result = get_dataset_info(df)

    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_values"] == 1


def test_unsupported_format(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("sample")

    with pytest.raises(DataAnalysisError):
        load_dataset(file_path)
