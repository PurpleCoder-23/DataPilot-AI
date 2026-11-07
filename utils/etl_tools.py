import os
import ipaddress
import socket
from urllib.parse import urlparse

import requests
import pandas as pd


class ETLTools:
    """
    ETL utilities for API extraction and dataset inspection/transformation.

    IMPORTANT:
    - API extraction is restricted to public HTTP(S) URLs.
    - LLM-generated Python code is NOT executed by this class.
      Use the constrained transformation methods below instead.
    """

    SUPPORTED_FORMATS = {"csv", "json", "parquet"}
    MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
    MAX_INPUT_BYTES = 50 * 1024 * 1024     # 50 MB

    def __init__(self):
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

    def _resolve_project_path(self, path: str) -> str:
        """Resolve a path inside the project root and reject path traversal."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("A valid file or folder path is required.")

        candidate = path.strip()
        if not os.path.isabs(candidate):
            candidate = os.path.join(self.project_root, candidate)

        resolved = os.path.abspath(candidate)

        try:
            common = os.path.commonpath([self.project_root, resolved])
        except ValueError:
            raise ValueError("Path is outside the project directory.")

        if common != self.project_root:
            raise ValueError(
                "For safety, ETL paths must stay inside the project directory."
            )

        return resolved

    @staticmethod
    def _validate_format(output_format: str) -> str:
        if not isinstance(output_format, str):
            raise ValueError("Output format must be a string.")

        output_format = output_format.lower().strip()
        if output_format not in ETLTools.SUPPORTED_FORMATS:
            raise ValueError(
                "Unsupported format. Choose csv, json, or parquet."
            )
        return output_format

    @staticmethod
    def _validate_public_http_url(url: str) -> str:
        """
        Basic SSRF protection for local development.
        Rejects localhost/private/reserved IPs and non-HTTP(S) schemes.
        Redirects are disabled so a public URL cannot redirect to a local host.
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("A valid API URL is required.")

        parsed = urlparse(url.strip())

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only HTTP and HTTPS API URLs are supported.")

        if not parsed.hostname:
            raise ValueError("The URL must include a hostname.")

        hostname = parsed.hostname.lower().rstrip(".")

        blocked_hostnames = {
            "localhost",
            "localhost.localdomain",
            "metadata.google.internal",
            "metadata",
        }

        if hostname in blocked_hostnames or hostname.endswith(".localhost"):
            raise ValueError("Localhost and metadata hostnames are not allowed.")

        try:
            addresses = {ipaddress.ip_address(hostname)}
        except ValueError:
            try:
                resolved_addresses = socket.getaddrinfo(
                    hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
                )
                addresses = {
                    ipaddress.ip_address(item[4][0].split("%")[0])
                    for item in resolved_addresses
                }
            except socket.gaierror as error:
                raise ValueError(f"Could not resolve API hostname: {error}")

        for address in addresses:
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
                or address.is_multicast
                or address.is_unspecified
            ):
                raise ValueError(
                    "Private, local, or reserved network addresses are not allowed."
                )

        return url.strip()

    def _read_dataset(self, file_path: str) -> pd.DataFrame:
        """Read a supported dataset after validating its project path and size."""
        resolved_path = self._resolve_project_path(file_path)

        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        file_size = os.path.getsize(resolved_path)
        if file_size > self.MAX_INPUT_BYTES:
            raise ValueError(
                f"Input file exceeds the {self.MAX_INPUT_BYTES // (1024 * 1024)} MB limit."
            )

        extension = os.path.splitext(resolved_path)[1].lower()

        if extension == ".csv":
            return pd.read_csv(resolved_path)
        if extension == ".json":
            # Supports both JSON Lines and regular JSON files.
            try:
                return pd.read_json(resolved_path, lines=True)
            except ValueError:
                return pd.read_json(resolved_path)
        if extension == ".parquet":
            return pd.read_parquet(resolved_path)

        raise ValueError(
            f"Unsupported input file type: {extension}. "
            "Supported: CSV, JSON, and Parquet."
        )

    def _save_dataset(
        self,
        df: pd.DataFrame,
        output_folder: str,
        output_format: str,
        filename_stem: str = "transformed_data",
    ) -> str:
        """Save a DataFrame to a project-contained destination."""
        output_format = self._validate_format(output_format)
        resolved_folder = self._resolve_project_path(output_folder)
        os.makedirs(resolved_folder, exist_ok=True)

        filename = os.path.join(
            resolved_folder, f"{filename_stem}.{output_format}"
        )

        if output_format == "csv":
            df.to_csv(filename, index=False)
        elif output_format == "json":
            df.to_json(filename, orient="records", lines=True)
        elif output_format == "parquet":
            df.to_parquet(filename, index=False)

        return filename

    def extract_load(self, url: str, output_folder: str, format: str):
        """Fetch a public JSON API and save the normalized result."""
        try:
            safe_url = self._validate_public_http_url(url)
            output_format = self._validate_format(format)
            resolved_folder = self._resolve_project_path(output_folder)

            with requests.get(
                safe_url,
                timeout=(5, 20),
                stream=True,
                allow_redirects=False,
                headers={"Accept": "application/json"},
            ) as response:
                response.raise_for_status()

                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > self.MAX_DOWNLOAD_BYTES:
                    return "Extraction failed: API response exceeds the 10 MB limit."

                chunks = []
                total_bytes = 0

                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > self.MAX_DOWNLOAD_BYTES:
                        return "Extraction failed: API response exceeds the 10 MB limit."
                    chunks.append(chunk)

            import json

            raw_data = b"".join(chunks)
            try:
                data = json.loads(raw_data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                return f"Extraction failed: API response is not valid JSON ({error})."

            df = pd.json_normalize(data)

            os.makedirs(resolved_folder, exist_ok=True)
            filename = os.path.join(
                resolved_folder, f"extracted_data.{output_format}"
            )

            if output_format == "csv":
                df.to_csv(filename, index=False)
            elif output_format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif output_format == "parquet":
                df.to_parquet(filename, index=False)

            return (
                f"Data successfully extracted and saved to {filename}\n"
                f"Total rows: {len(df)}\n"
                f"Total columns: {len(df.columns)}"
            )

        except requests.exceptions.RequestException as error:
            return f"Failed to extract data: {type(error).__name__}: {error}"
        except (ValueError, OSError, ImportError) as error:
            return f"Failed to process or save data: {type(error).__name__}: {error}"

    def transform_load_context(
        self,
        file_path: str,
        output_folder: str,
        output_format: str,
    ):
        """Return dataset metadata and a small preview for the transformation LLM."""
        try:
            output_format = self._validate_format(output_format)
            df = self._read_dataset(file_path)

            preview_columns = list(df.columns[:20])
            preview = df[preview_columns].head(3)

            return (
                f"Input file: {self._resolve_project_path(file_path)}\n"
                f"Rows: {len(df)}\n"
                f"Columns count: {len(df.columns)}\n"
                f"Column names: {list(df.columns)}\n"
                f"Requested output folder: {output_folder}\n"
                f"Requested output format: {output_format}\n"
                f"Sample rows:\n{preview.to_string(index=False, max_colwidth=30)}"
            )

        except Exception as error:
            return (
                f"Failed to read input file: "
                f"{type(error).__name__}: {error}"
            )

    def transform_dataset(
        self,
        input_file_path: str,
        output_folder: str,
        output_format: str = "csv",
        select_columns=None,
        drop_columns=None,
        filter_column=None,
        filter_operator=None,
        filter_value=None,
        sort_by=None,
        ascending=True,
        drop_duplicates=False,
        limit=None,
    ):
        """
        Apply a limited, explicit set of dataframe operations.

        This replaces arbitrary LLM-generated Python execution. The LLM should
        produce structured arguments for these operations, not Python code.
        """
        try:
            df = self._read_dataset(input_file_path)

            if select_columns:
                if isinstance(select_columns, str):
                    select_columns = [select_columns]
                missing = [col for col in select_columns if col not in df.columns]
                if missing:
                    return f"Transformation failed: columns not found: {missing}"
                df = df.loc[:, select_columns]

            if drop_columns:
                if isinstance(drop_columns, str):
                    drop_columns = [drop_columns]
                missing = [col for col in drop_columns if col not in df.columns]
                if missing:
                    return f"Transformation failed: columns not found: {missing}"
                df = df.drop(columns=drop_columns)

            if filter_column is not None:
                if filter_column not in df.columns:
                    return f"Transformation failed: column not found: {filter_column}"

                operators = {
                    "equals": lambda series, value: series == value,
                    "not_equals": lambda series, value: series != value,
                    "greater_than": lambda series, value: series > value,
                    "greater_than_or_equal": lambda series, value: series >= value,
                    "less_than": lambda series, value: series < value,
                    "less_than_or_equal": lambda series, value: series <= value,
                    "contains": lambda series, value: series.astype(str).str.contains(
                        str(value), case=False, na=False, regex=False
                    ),
                    "is_null": lambda series, value: series.isna(),
                    "not_null": lambda series, value: series.notna(),
                }

                if filter_operator not in operators:
                    return (
                        "Transformation failed: unsupported filter operator. "
                        f"Choose one of {list(operators)}."
                    )

                try:
                    mask = operators[filter_operator](
                        df[filter_column], filter_value
                    )
                    df = df.loc[mask]
                except (TypeError, ValueError) as error:
                    return f"Transformation failed while filtering: {error}"

            if sort_by:
                if isinstance(sort_by, str):
                    sort_by = [sort_by]
                missing = [col for col in sort_by if col not in df.columns]
                if missing:
                    return f"Transformation failed: sort columns not found: {missing}"
                df = df.sort_values(by=sort_by, ascending=ascending)

            if drop_duplicates:
                df = df.drop_duplicates()

            if limit is not None:
                if not isinstance(limit, int) or limit < 1:
                    return "Transformation failed: limit must be a positive integer."
                df = df.head(limit)

            output_path = self._save_dataset(
                df=df,
                output_folder=output_folder,
                output_format=output_format,
            )

            return (
                "TRANSFORMATION_SUCCESS\n"
                f"Output file: {output_path}\n"
                f"Rows after transformation: {len(df)}\n"
                f"Columns after transformation: {len(df.columns)}\n"
                f"Columns: {list(df.columns)}"
            )

        except Exception as error:
            return (
                f"TRANSFORMATION_FAILED: "
                f"{type(error).__name__}: {error}"
            )



    # ---------------------------------------------------------
    # In-memory transformation support for uploaded datasets
    # ---------------------------------------------------------

    @staticmethod
    def transform_dataframe(
        dataframe: pd.DataFrame,
        select_columns=None,
        drop_columns=None,
        filter_column=None,
        filter_operator=None,
        filter_value=None,
        sort_by=None,
        ascending=True,
        drop_duplicates=False,
        limit=None,
    ) -> pd.DataFrame:
        """Transform an uploaded/in-memory DataFrame without modifying it.

        This method deliberately accepts a DataFrame instead of a local path,
        allowing Streamlit uploads to be processed directly.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")

        df = dataframe.copy()

        if select_columns:
            if isinstance(select_columns, str):
                select_columns = [select_columns]
            missing = [col for col in select_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Selected columns not found: {missing}")
            df = df.loc[:, select_columns].copy()

        if drop_columns:
            if isinstance(drop_columns, str):
                drop_columns = [drop_columns]
            missing = [col for col in drop_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Columns to drop not found: {missing}")
            df = df.drop(columns=drop_columns)

        if filter_column is not None:
            if filter_column not in df.columns:
                raise ValueError(f"Filter column not found: {filter_column}")

            operators = {
                "equals": lambda s, v: s == v,
                "not_equals": lambda s, v: s != v,
                "greater_than": lambda s, v: s > v,
                "greater_than_or_equal": lambda s, v: s >= v,
                "less_than": lambda s, v: s < v,
                "less_than_or_equal": lambda s, v: s <= v,
                "contains": lambda s, v: s.astype("string").str.contains(
                    str(v), case=False, na=False, regex=False
                ),
                "startswith": lambda s, v: s.astype("string").str.startswith(
                    str(v), na=False
                ),
                "endswith": lambda s, v: s.astype("string").str.endswith(
                    str(v), na=False
                ),
                "is_null": lambda s, v: s.isna(),
                "not_null": lambda s, v: s.notna(),
            }

            if filter_operator not in operators:
                raise ValueError(
                    "Unsupported filter operator. Choose: "
                    + ", ".join(operators.keys())
                )

            try:
                mask = operators[filter_operator](df[filter_column], filter_value)
                df = df.loc[mask].copy()
            except (TypeError, ValueError) as error:
                raise ValueError(f"Filtering failed: {error}") from error

        if sort_by:
            sort_columns = [sort_by] if isinstance(sort_by, str) else list(sort_by)
            missing = [col for col in sort_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Sort columns not found: {missing}")

            df = df.sort_values(
                by=sort_columns,
                ascending=ascending,
                na_position="last",
            )

        if drop_duplicates:
            df = df.drop_duplicates()

        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError("limit must be a non-negative integer.")
            df = df.head(limit)

        return df.reset_index(drop=True)

    @staticmethod
    def dataframe_to_bytes(
        dataframe: pd.DataFrame,
        output_format: str = "csv",
    ) -> tuple[bytes, str, str]:
        """Serialize a DataFrame for Streamlit's download_button.

        Returns (file_bytes, MIME type, suggested filename).
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")

        output_format = ETLTools._validate_format(output_format)

        if output_format == "csv":
            payload = dataframe.to_csv(index=False).encode("utf-8-sig")
            mime_type = "text/csv"
            filename = "transformed_data.csv"

        elif output_format == "json":
            payload = dataframe.to_json(
                orient="records",
                lines=True,
                force_ascii=False,
            ).encode("utf-8")
            mime_type = "application/json"
            filename = "transformed_data.json"

        else:
            try:
                payload_buffer = __import__("io").BytesIO()
                dataframe.to_parquet(payload_buffer, index=False)
                payload = payload_buffer.getvalue()
            except ImportError as error:
                raise ImportError(
                    "Parquet export requires pyarrow or fastparquet."
                ) from error

            mime_type = "application/octet-stream"
            filename = "transformed_data.parquet"

        return payload, mime_type, filename

    def execute_code(self, code: str):
        """
        Deprecated compatibility method.

        Arbitrary Python execution has intentionally been disabled because
        executing LLM-generated code with exec() is not a security sandbox.
        """
        return (
            "EXECUTION_BLOCKED: Arbitrary Python code execution is disabled. "
            "Use transform_dataset() with supported transformation operations."
        )


if __name__ == "__main__":
    etl_tools = ETLTools()

    test_file = "data/extract/extracted_data.csv"
    print(
        etl_tools.transform_load_context(
            test_file,
            "data/transform",
            "csv",
        )
    )
