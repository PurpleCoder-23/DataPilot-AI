import pandas as pd
import pytest

from utils.data_analysis import (
    convert_column_types,
    drop_columns,
    handle_missing_values,
    remove_duplicates,
    rename_columns,
)


# ---------------------------------------------------------
# Sample dataset fixture
# ---------------------------------------------------------

@pytest.fixture
def sample_df():

    return pd.DataFrame({
        "name": ["A", "B", "B", "D"],
        "age": [20, None, None, 40],
        "city": ["Hyd", "Delhi", "Delhi", None],
    })


# ---------------------------------------------------------
# Test duplicate removal
# ---------------------------------------------------------

def test_remove_duplicates(sample_df):

    result = remove_duplicates(sample_df)

    assert len(result) == 3

    # Original dataset must remain unchanged
    assert len(sample_df) == 4


# ---------------------------------------------------------
# Test missing value removal
# ---------------------------------------------------------

def test_handle_missing_drop_rows(sample_df):

    result = handle_missing_values(
        sample_df,
        strategy="drop_rows"
    )

    assert result.isna().sum().sum() == 0

    assert len(result) == 1


# ---------------------------------------------------------
# Test custom missing value filling
# ---------------------------------------------------------

def test_handle_missing_fill_value(sample_df):

    result = handle_missing_values(
        sample_df,
        strategy="fill_value",
        fill_value="Unknown",
        columns=["city"]
    )

    assert result["city"].isna().sum() == 0

    # Original dataset must remain unchanged
    assert sample_df["city"].isna().sum() == 1


# ---------------------------------------------------------
# Test mean imputation
# ---------------------------------------------------------

def test_handle_missing_mean(sample_df):

    result = handle_missing_values(
        sample_df,
        strategy="mean",
        columns=["age"]
    )

    assert result["age"].isna().sum() == 0

    # Mean of 20 and 40 = 30
    assert result["age"].iloc[1] == 30


# ---------------------------------------------------------
# Test dropping columns
# ---------------------------------------------------------

def test_drop_columns(sample_df):

    result = drop_columns(
        sample_df,
        ["city"]
    )

    assert "city" not in result.columns

    # Original dataset must remain unchanged
    assert "city" in sample_df.columns


# ---------------------------------------------------------
# Test renaming columns
# ---------------------------------------------------------

def test_rename_columns(sample_df):

    result = rename_columns(
        sample_df,
        {"name": "person"}
    )

    assert "person" in result.columns

    # Original dataset must remain unchanged
    assert "name" in sample_df.columns


# ---------------------------------------------------------
# Test duplicate column name protection
# ---------------------------------------------------------

def test_rename_rejects_duplicate_names(sample_df):

    with pytest.raises(
        ValueError,
        match="duplicate column names"
    ):

        rename_columns(
            sample_df,
            {"name": "age"}
        )


# ---------------------------------------------------------
# Test data type conversion
# ---------------------------------------------------------

def test_convert_column_types(sample_df):

    result = convert_column_types(
        sample_df,
        {"age": "float"}
    )

    assert pd.api.types.is_float_dtype(
        result["age"]
    )


# ---------------------------------------------------------
# Test unknown column handling
# ---------------------------------------------------------

def test_cleaning_rejects_unknown_column(sample_df):

    with pytest.raises(
        ValueError,
        match="Unknown column"
    ):

        drop_columns(
            sample_df,
            ["not_a_column"]
        )