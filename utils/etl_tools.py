
import os
import requests
import pandas as pd


class ETLTools:

    def __init__(self):
        pass

    def extract_load(self, url: str, output_folder: str, format: str):
        """
        Extract data from an API and save it to the desired destination.

        Args:
            url (str): API endpoint from which to extract data.
            output_folder (str): Destination folder for extracted data.
            format (str): Output format: csv, json, or parquet.

        Returns:
            str: A message indicating success or failure.
        """

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        output_folder = os.path.join(project_root, output_folder)
        output_format = format.lower().strip()

        try:
            # Step 1: Send the API request.
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Step 2: Parse the API response as JSON.
            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                return (
                    "Failed to extract data: API response is not valid JSON. "
                    f"Response preview: {response.text[:200]!r}"
                )

            # Step 3: Convert JSON data into a Pandas DataFrame.
            df = pd.json_normalize(data)

            # Step 4: Create the destination directory.
            os.makedirs(output_folder, exist_ok=True)

            # Step 5: Create the output filename.
            filename = os.path.join(
                output_folder,
                f"extracted_data.{output_format}"
            )

            # Step 6: Save data in the requested format.
            if output_format == "csv":
                df.to_csv(filename, index=False)

            elif output_format == "json":
                df.to_json(
                    filename,
                    orient="records",
                    lines=True
                )

            elif output_format == "parquet":
                df.to_parquet(filename, index=False)

            else:
                return (
                    f"Unsupported format: {format}. "
                    "Supported formats are csv, json, and parquet."
                )

            return (
                "Data successfully extracted and saved to "
                f"{filename}\n"
                f"Total rows: {len(df)}\n"
                f"Total columns: {len(df.columns)}"
            )

        except requests.exceptions.RequestException as e:
            return f"Failed to extract data: {e}"

        except (ValueError, OSError) as e:
            return f"Failed to process or save data: {e}"

        except ImportError as e:
            return f"Required dependency is missing: {e}"

    def transform_load_context(
        self,
        file_path: str,
        output_folder: str,
        output_format: str
    ):
        """
        Read the input dataset and return its first three rows
        as context for the LLM.

        Args:
            file_path (str): Path to the input dataset.
            output_folder (str): Destination folder for transformed data.
            output_format (str): Desired output format.

        Returns:
            str: The first three rows or an error message.
        """

        try:
            # Step 1: Check whether the input file exists.
            if not os.path.isfile(file_path):
                return f"Input file not found: {file_path}"

            # Step 2: Identify the input file format.
            file_extension = os.path.splitext(file_path)[1].lower()

            # Step 3: Read the input dataset.
            if file_extension == ".csv":
                df = pd.read_csv(file_path)

            elif file_extension == ".json":
                df = pd.read_json(file_path, lines=True)

            elif file_extension == ".parquet":
                df = pd.read_parquet(file_path)

            else:
                return (
                    f"Unsupported input file format: {file_extension}. "
                    "Supported formats are csv, json, and parquet."
                )

            # Step 4: Return sample rows for the LLM prompt.
            # Keep only a small sample of columns for the LLM.
            preview_columns = list(df.columns[:20])

            preview = df[preview_columns].head(2)

            return (
            f"Total rows: {len(df)}\n"
            f"Total columns: {len(df.columns)}\n"
            f"Column names: {list(df.columns)}\n"
            f"Sample data:\n{preview.to_string(index=False, max_colwidth=30)}"
        )

        except Exception as e:
            return (
                f"Failed to read input file: "
                f"{type(e).__name__}: {e}"
            )

    def execute_code(self, code: str):
        """
        Execute the generated transformation code.

        Args:
            code (str): Python code generated by the LLM.

        Returns:
            str: Execution status or an error message.
        """

        try:
            # Provide commonly used objects to the generated code.
            exec_globals = {
                "os": os,
                "pd": pd
            }

            # Step 1: Execute the generated Python code.
            exec(code, exec_globals)

            # Step 2: Return the execution status.
            return "Transformation code executed successfully."

        except Exception as e:
            return (
                f"Failed to execute code: "
                f"{type(e).__name__}: {e}"
            )


# ---------------------------------------------------------
# TEST THE ETL TOOL
# ---------------------------------------------------------

if __name__ == "__main__":

    obj = ETLTools()

    # Uncomment this block if you want to test extraction again.
    #
    # result = obj.extract_load(
    #     url="https://pokeapi.co/api/v2/pokemon/pikachu",
    #     output_folder="data/extract",
    #     format="csv"
    # )
    # print(result)

    # Test reading the extracted dataset.
    path = (
        r"C:\Users\HOME\Documents\DATA AGENT"
        r"\data\extract\extracted_data.csv"
    )

    result = obj.transform_load_context(
        path,
        r"C:\Users\HOME\Documents\DATA AGENT\data\transform",
        "csv"
    )

    print(result)
