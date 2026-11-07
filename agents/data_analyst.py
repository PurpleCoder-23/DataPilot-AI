"""
Data Analyst for uploaded datasets.

The LLM interprets a natural-language question as a constrained analysis plan.
Pandas performs the calculation locally on the uploaded DataFrame.
No arbitrary Python code is generated or executed.
"""

from __future__ import annotations

from typing import Literal, Optional, List

import re
import pandas as pd
from pydantic import BaseModel, Field

from utils.llm_pick import pick_llm


class DatasetQuestionPlan(BaseModel):
    """A safe, constrained plan for answering a dataset question."""

    operation: Literal[
        "row_count",
        "column_names",
        "select_columns",
        "sort_values",
        "preview",
        "missing_values",
        "duplicate_count",
        "describe_column",
        "value_counts",
        "correlation",
        "sql_query",
    ] = Field(description="The supported analysis operation.")

    column: Optional[str] = Field(
        default=None,
        description="A column name required for column-specific operations.",
    )
    second_column: Optional[str] = Field(
        default=None,
        description="Second numeric column for correlation.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Multiple exact dataset column names for SQL select_columns. Use this for requests to select more than one column.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of rows or categories to return.",
    )
    sql_action: Literal[
        "distinct_values",
        "group_count",
        "select_columns",
        "preview_rows",
        "select_all",
        "preview_all_columns",
    ] = Field(
        default="distinct_values",
        description="SQL query pattern when operation is sql_query.",
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Optional SQL table name; use a safe placeholder if not provided.",
    )


class DataAnalystError(Exception):
    """Raised when a dataset question cannot be safely answered."""


def _format_number(value) -> str:
    """Format common numeric values cleanly."""
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int,)):
        return str(value)

    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.4f}"
    except (TypeError, ValueError):
        return str(value)


def _quote_sql_identifier(identifier: str) -> str:
    """Quote a SQL Server identifier using brackets."""
    return "[" + str(identifier).replace("]", "]]") + "]"


def _generate_sql_from_plan(
    df: pd.DataFrame,
    plan: DatasetQuestionPlan,
) -> str:
    """Generate a safe SQL Server SELECT query from the uploaded dataset schema."""
    table_name = (plan.table_name or "").strip() or "YourTable"
    parts = table_name.split(".")
    if len(parts) not in (1, 2) or not all(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", part.strip()) for part in parts
    ):
        raise DataAnalystError(
            "Please provide a simple table name, such as Students or dbo.Students."
        )

    quoted_table = ".".join(_quote_sql_identifier(part.strip()) for part in parts)

    if plan.sql_action in {"select_all", "preview_all_columns"}:
        if plan.sql_action == "select_all":
            query = f"SELECT *\nFROM {quoted_table};"
            explanation = "Returns all columns and rows from the table."
        else:
            query = f"SELECT TOP ({plan.limit}) *\nFROM {quoted_table};"
            explanation = f"Returns the first {plan.limit} rows and all columns."
    else:
        requested_columns = plan.columns or ([plan.column] if plan.column else [])
        valid_columns = []
        for requested in requested_columns:
            if requested not in df.columns:
                matches = [
                    c for c in df.columns
                    if str(c).casefold() == str(requested).casefold()
                ]
                if len(matches) == 1:
                    requested = matches[0]
                else:
                    raise DataAnalystError(
                        f"Column '{requested}' was not found. Available columns: {list(df.columns)}"
                    )
            if requested not in valid_columns:
                valid_columns.append(requested)

        if plan.sql_action in {"distinct_values", "group_count", "select_columns", "preview_rows"} and not valid_columns:
            raise DataAnalystError(
                "Please specify a column. Available columns: " + ", ".join(map(str, df.columns))
            )

        if plan.sql_action == "distinct_values":
            column = valid_columns[0]
            quoted_column = _quote_sql_identifier(str(column))
            query = (
                f"SELECT DISTINCT {quoted_column}\n"
                f"FROM {quoted_table}\n"
                f"ORDER BY {quoted_column};"
            )
            explanation = f"Returns distinct values in '{column}'."
        elif plan.sql_action == "group_count":
            column = valid_columns[0]
            quoted_column = _quote_sql_identifier(str(column))
            query = (
                f"SELECT {quoted_column}, COUNT(*) AS [RecordCount]\n"
                f"FROM {quoted_table}\n"
                f"GROUP BY {quoted_column}\n"
                f"ORDER BY [RecordCount] DESC;"
            )
            explanation = f"Counts records for each value in '{column}'."
        elif plan.sql_action == "select_columns":
            quoted_columns = ", ".join(_quote_sql_identifier(str(c)) for c in valid_columns)
            query = f"SELECT {quoted_columns}\nFROM {quoted_table};"
            explanation = "Returns selected column(s): " + ", ".join(map(str, valid_columns)) + "."
        elif plan.sql_action == "preview_rows":
            quoted_columns = ", ".join(_quote_sql_identifier(str(c)) for c in valid_columns)
            query = f"SELECT TOP ({plan.limit}) {quoted_columns}\nFROM {quoted_table};"
            explanation = f"Returns up to {plan.limit} rows for selected column(s)."
        else:
            raise DataAnalystError(f"Unsupported SQL generation action: {plan.sql_action}")

    return (
        "Generated SQL Server query (not executed):\n\n"
        f"```sql\n{query}\n```\n\n"
        f"{explanation}\n\n"
        f"**Note:** Replace `{table_name}` with the actual SQL table name. "
        "This query is generated from your uploaded dataset's columns; "
        "it has not been executed against SQL Server."
    )


def _execute_plan(df: pd.DataFrame, plan: DatasetQuestionPlan) -> str:
    """Execute the validated plan against the supplied DataFrame."""

    if df is None:
        raise DataAnalystError("No dataset was provided.")

    if not isinstance(df, pd.DataFrame):
        raise DataAnalystError("The uploaded data is not a pandas DataFrame.")

    if plan.operation == "sql_query":
        return _generate_sql_from_plan(df, plan)

    if df.empty and plan.operation not in {"column_names", "row_count"}:
        return "The uploaded dataset is empty."

    if plan.operation == "row_count":
        return (
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns):,}"
        )

    if plan.operation == "column_names":
        if not len(df.columns):
            return "The dataset has no columns."
        return "Columns:\n" + "\n".join(
            f"- {column}" for column in df.columns
        )

    if plan.operation == "select_columns":
        requested_columns = plan.columns or ([plan.column] if plan.column else [])
        if not requested_columns:
            raise DataAnalystError(
                "Please specify one or more columns. Available columns: "
                + ", ".join(map(str, df.columns))
            )
        selected = []
        for requested in requested_columns:
            if requested not in df.columns:
                matches = [
                    c for c in df.columns
                    if str(c).casefold() == str(requested).casefold()
                ]
                if len(matches) == 1:
                    requested = matches[0]
                else:
                    raise DataAnalystError(
                        f"Column '{requested}' was not found. "
                        f"Available columns: {list(df.columns)}"
                    )
            if requested not in selected:
                selected.append(requested)
        return df.loc[:, selected].head(plan.limit).to_string(index=False)

    if plan.operation == "sort_values":
        requested_columns = plan.columns or ([plan.column] if plan.column else [])
        if not requested_columns:
            raise DataAnalystError(
                "Please specify the column(s) to sort by. "
                f"Available columns: {list(df.columns)}"
            )
        sort_columns = []
        for requested in requested_columns:
            if requested not in df.columns:
                matches = [
                    c for c in df.columns
                    if str(c).casefold() == str(requested).casefold()
                ]
                if len(matches) == 1:
                    requested = matches[0]
                else:
                    raise DataAnalystError(
                        f"Column '{requested}' was not found. "
                        f"Available columns: {list(df.columns)}"
                    )
            sort_columns.append(requested)
        sorted_df = df.sort_values(
            by=sort_columns,
            ascending=plan.ascending,
            kind="stable",
            na_position="last",
        )
        return sorted_df.head(plan.limit).to_string(index=False)

    if plan.operation == "preview":
        return df.head(plan.limit).to_string(index=False)

    if plan.operation == "missing_values":
        missing = df.isna().sum()
        total_missing = int(missing.sum())

        lines = [
            f"Total missing cells: {total_missing:,}",
            "",
            "Missing values by column:",
        ]

        for column, count in missing.items():
            lines.append(f"- {column}: {int(count):,}")

        return "\n".join(lines)

    if plan.operation == "duplicate_count":
        count = int(df.duplicated().sum())
        return f"Duplicate rows: {count:,}"

    if plan.operation == "describe_column":
        column = plan.column

        if column not in df.columns:
            raise DataAnalystError(
                f"Column '{column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )

        series = df[column]
        lines = [
            f"Column: {column}",
            f"Data type: {series.dtype}",
            f"Non-null values: {int(series.count()):,}",
            f"Missing values: {int(series.isna().sum()):,}",
            f"Unique values: {int(series.nunique(dropna=True)):,}",
        ]

        if pd.api.types.is_numeric_dtype(series):
            lines.extend(
                [
                    f"Mean: {_format_number(series.mean())}",
                    f"Median: {_format_number(series.median())}",
                    f"Minimum: {_format_number(series.min())}",
                    f"Maximum: {_format_number(series.max())}",
                    f"Standard deviation: {_format_number(series.std())}",
                ]
            )
        else:
            mode = series.mode(dropna=True)
            if not mode.empty:
                lines.append(f"Most frequent value: {mode.iloc[0]}")

        return "\n".join(lines)

    if plan.operation == "value_counts":
        column = plan.column

        if column not in df.columns:
            raise DataAnalystError(
                f"Column '{column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )

        counts = df[column].value_counts(dropna=False).head(plan.limit)
        return (
            f"Top {min(plan.limit, len(counts))} values in '{column}':\n"
            + counts.to_string()
        )

    if plan.operation == "correlation":
        first = plan.column
        second = plan.second_column

        if first not in df.columns or second not in df.columns:
            raise DataAnalystError(
                "Both columns must exist. "
                f"Available columns: {list(df.columns)}"
            )

        if not pd.api.types.is_numeric_dtype(df[first]):
            raise DataAnalystError(f"'{first}' is not numeric.")

        if not pd.api.types.is_numeric_dtype(df[second]):
            raise DataAnalystError(f"'{second}' is not numeric.")

        correlation = df[first].corr(df[second])
        return (
            f"Pearson correlation between '{first}' and '{second}': "
            f"{_format_number(correlation)}"
        )

    raise DataAnalystError(
        f"Unsupported analysis operation: {plan.operation}"
    )



def _generate_five_dataset_sql_queries(df: pd.DataFrame) -> str:
    """Generate five useful SQL examples using only columns present in df."""
    cols = list(df.columns)
    if not cols:
        return "The uploaded dataset has no columns to build SQL queries from."

    table = "[YourTable]"
    qcol = lambda c: _quote_sql_identifier(str(c))
    queries = []

    # Query 1: all records
    queries.append((
        f"SELECT *\nFROM {table};",
        "Returns every row and column from the table."
    ))

    # Query 2: sort by a likely human-readable column, otherwise first column
    name_col = next((c for c in cols if str(c).strip().casefold() == "name"), cols[0])
    queries.append((
        f"SELECT *\nFROM {table}\nORDER BY {qcol(name_col)} ASC;",
        f"Sorts all records alphabetically/by value using {qcol(name_col)}."
    ))

    # Query 3: count by category-like column, if available
    category_col = next(
        (c for c in cols if str(c).strip().casefold() in
         {"technology", "category", "department", "course", "batch number"}),
        cols[0]
    )
    queries.append((
        f"SELECT {qcol(category_col)}, COUNT(*) AS [RecordCount]\n"
        f"FROM {table}\nGROUP BY {qcol(category_col)}\n"
        "ORDER BY [RecordCount] DESC;",
        f"Counts records in each {qcol(category_col)} group, highest count first."
    ))

    # Query 4: select a few meaningful fields (first up to 3 actual columns)
    selected = cols[:min(3, len(cols))]
    selected_sql = ", ".join(qcol(c) for c in selected)
    queries.append((
        f"SELECT {selected_sql}\nFROM {table};",
        "Returns only these dataset columns: " + ", ".join(map(str, selected)) + "."
    ))

    # Query 5: count rows overall
    queries.append((
        f"SELECT COUNT(*) AS [TotalRows]\nFROM {table};",
        "Returns the total number of records in the table."
    ))

    sections = ["## Five SQL Server queries for your uploaded dataset"]
    for i, (query, explanation) in enumerate(queries, start=1):
        sections.append(
            f"### Query {i}\n```sql\n{query}\n```\n**Explanation:** {explanation}"
        )
    sections.append(
        "**Note:** Replace `[YourTable]` with your real SQL Server table name. "
        "These are examples based on your uploaded dataset's column names; "
        "they are not executed against SQL Server."
    )
    return "\n\n".join(sections)


def _generate_trim_sql(df: pd.DataFrame) -> str:
    """Generate SELECT-only SQL that trims leading/trailing spaces from text columns."""
    text_columns = [
        col for col in df.columns
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object
    ]
    if not text_columns:
        return (
            "No text columns were detected in the uploaded dataset. "
            "There are no string values to trim."
        )

    expressions = []
    for col in df.columns:
        quoted = _quote_sql_identifier(str(col))
        if col in text_columns:
            expressions.append(
                f"LTRIM(RTRIM({quoted})) AS {quoted}"
            )
        else:
            expressions.append(quoted)

    query = (
        "SELECT\n    " + ",\n    ".join(expressions) +
        "\nFROM [YourTable];"
    )
    return (
        "## SQL Server query to remove leading/trailing spaces from text values\n\n"
        f"```sql\n{query}\n```\n\n"
        "**Explanation:** `LTRIM` removes leading spaces and `RTRIM` removes "
        "trailing spaces from text columns. This SELECT previews cleaned values "
        "without modifying the source table. To permanently update the table, "
        "you would need a separate UPDATE statement and should back up the data first.\n\n"
        "**Note:** Replace `[YourTable]` with your actual SQL Server table name. "
        "This query is generated from your uploaded dataset and is not executed."
    )


def answer_dataset_question(
    df: pd.DataFrame,
    question: str,
) -> str:
    """
    Answer a natural-language question about an uploaded DataFrame.

    Supported questions include:
    - How many rows and columns are there?
    - What are the column names?
    - Select multiple columns or sort rows by a column.
    - Show me the first few rows.
    - How many missing values are in each column?
    - How many duplicate rows are there?
    - Give me statistics for a column.
    - What are the most common values in a column?
    - What is the correlation between two numeric columns?
    """

    if df is None:
        return "Please upload a dataset first."

    if not isinstance(df, pd.DataFrame):
        return "The uploaded dataset is not in a supported DataFrame format."

    if not isinstance(question, str) or not question.strip():
        return "Please enter a question about your uploaded dataset."

    # Deterministic handling for SQL requests that do not need a column.
    normalized_question = " ".join(question.lower().split())
    first_n_match = re.search(
        r"\b(?:first|top)\s+(\d+)\s+rows?\b",
        normalized_question,
    )
    # Deterministic natural-language handling for common SQL requests.
    if (
        ("generate" in normalized_question or "give" in normalized_question
         or "write" in normalized_question or "create" in normalized_question)
        and re.search(r"\b5\b|\bfive\b", normalized_question)
        and ("sql" in normalized_question or "queries" in normalized_question)
    ):
        return _generate_five_dataset_sql_queries(df)

    if (
        ("remove space" in normalized_question or "trim" in normalized_question
         or "leading and trailing spaces" in normalized_question)
        and ("column" in normalized_question or "columns" in normalized_question
             or "values" in normalized_question)
        and ("sql" in normalized_question or "query" in normalized_question)
    ):
        return _generate_trim_sql(df)

    # Deterministic ORDER BY query generation using actual uploaded columns.
    if ("sql" in normalized_question or "query" in normalized_question) and (
        "sort" in normalized_question or "order by" in normalized_question
        or "ascending" in normalized_question or "descending" in normalized_question
    ):
        matched_col = None
        for col in df.columns:
            col_name = str(col).strip()
            if col_name and re.search(
                r"(?<![\w])" + re.escape(col_name.lower()) + r"(?![\w])",
                normalized_question
            ):
                matched_col = col
                break
        if matched_col is not None:
            direction = "DESC" if any(
                term in normalized_question for term in
                ("descending", "desc", "z to a", "highest to lowest")
            ) else "ASC"
            query = (
                "SELECT *\nFROM [YourTable]\n"
                f"ORDER BY {_quote_sql_identifier(str(matched_col).strip())} {direction};"
            )
            return (
                "Generated SQL query (not executed):\n\n"
                f"```sql\n{query}\n```\n\n"
                f"Sorts rows by {_quote_sql_identifier(str(matched_col).strip())} "
                f"in {direction.lower()} order.\n\n"
                "Note: Replace `[YourTable]` with your actual SQL table name. "
                "This query uses your uploaded dataset's columns and has not been executed."
            )

    if "sql" in normalized_question or "query" in normalized_question:
        if first_n_match:
            limit = max(1, min(int(first_n_match.group(1)), 50))
            plan = DatasetQuestionPlan(
                operation="sql_query",
                column=None,
                limit=limit,
                sql_action="preview_all_columns",
            )
            return _execute_plan(df, plan)

        if any(
            phrase in normalized_question
            for phrase in (
                "all records", "all rows", "every record", "every row",
                "display all", "show all records", "select all",
            )
        ):
            plan = DatasetQuestionPlan(
                operation="sql_query",
                column=None,
                sql_action="select_all",
            )
            return _execute_plan(df, plan)

    try:
        llm = pick_llm("high")
        planner = llm.with_structured_output(
            DatasetQuestionPlan,
            method="json_schema",
        )

        # Only column names and the question are sent to the LLM.
        # Dataset rows and values remain local.
        prompt = f"""
You are a careful data analyst. Convert the user's question into
one supported analysis plan.

Available dataset columns:
{list(df.columns)}

User question:
{question}

Choose only one operation from the schema:
- row_count: number of rows and columns
- column_names: list column names
- select_columns: return selected column(s) from the uploaded data. Put all
  requested exact names in columns, preserving their order.
- sort_values: sort uploaded data by one or more columns. Put sort columns in
  columns and set ascending=false for descending order.
- preview: show a few beginning rows
- missing_values: missing-value counts by column
- duplicate_count: count duplicate rows
- describe_column: statistics for one specified column
- value_counts: most frequent values for one specified column
- correlation: Pearson correlation between two numeric columns
- sql_query: generate a SQL Server SELECT query based on the uploaded file's
  column names. Use sql_action:
  * distinct_values for "different/distinct/unique values"
  * group_count for counts by category/technology
  * select_columns for selecting one or more named columns
  * preview_rows for showing a few values from one or more columns
  * preview_all_columns for first N rows with all columns
  * select_all for all rows and all columns
  For preview_all_columns and select_all, column should be null.
  For other SQL actions, use the relevant exact dataset column.
  For select_columns or preview_rows with multiple requested columns, populate
  the columns list with each exact column name in the user's requested order.
  For single-column operations, set column to the exact column name.
  If the user does not give a table name, leave table_name null.

Rules:
- Use exact column names from the available columns.
- For "select Technology and Name", put both exact names in the columns list,
  preserving the requested order.
- For sorting, use the exact sort column(s) and infer ascending/descending only
  from the user's wording.
- Do not invent column names.
- If the question is ambiguous, choose the closest supported operation.
- For preview or value_counts, choose a reasonable limit up to 50.
"""

        plan = planner.invoke(prompt)

        # Validate column names and execute locally with pandas.
        return _execute_plan(df, plan)

    except DataAnalystError as error:
        return f"Data Analyst: {error}"

    except Exception as error:
        print(
            "Data Analyst error: "
            f"{type(error).__name__}: {error}"
        )
        return (
            "I couldn't analyze that question. Please try asking about "
            "row counts, columns, missing values, duplicates, column "
            "statistics, value counts, or correlation."
        )
