
import pyodbc


class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

        # Build the SQL Server connection string.
        driver = db_config["DRIVER"].strip("{}")

        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={db_config['SERVER']};"
            f"DATABASE={db_config['DATABASE']};"
            f"Trusted_Connection={db_config['Trusted_Connection']};"
            f"TrustServerCertificate={db_config['TrustServerCertificate']};"
        )

        try:
            self.connection = pyodbc.connect(connection_string)
            print("Connected to SQL Server successfully.")

        except pyodbc.Error as e:
            print(f"Error connecting to SQL Server: {e}")
            raise

    @staticmethod
    def quote_identifier(name):
        """Safely quote a SQL Server schema or table name."""
        return "[" + name.replace("]", "]]") + "]"

    def schema_details(self, schema_name="dbo"):
        """Return table names, columns, data types, and sample rows."""

        if self.connection is None:
            raise RuntimeError("Database connection is not available.")

        cursor = None

        try:
            cursor = self.connection.cursor()
            schema_info_context = f"Database Schema: {schema_name}\n"

            cursor.execute(
                """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = ?
                  AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
                """,
                (schema_name,)
            )

            tables_list = cursor.fetchall()

            if not tables_list:
                schema_info_context += (
                    f"\nNo tables found in schema '{schema_name}'.\n"
                )

            for (table_name,) in tables_list:
                schema_info_context += f"\nTable: {table_name}\n"

                # Get column names and data types.
                cursor.execute(
                    """
                    SELECT COLUMN_NAME, DATA_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = ?
                      AND TABLE_SCHEMA = ?
                    ORDER BY ORDINAL_POSITION
                    """,
                    (table_name, schema_name)
                )

                for column_name, data_type in cursor.fetchall():
                    schema_info_context += (
                        f"  Column: {column_name}, "
                        f"Data Type: {data_type}\n"
                    )

                # Get up to five sample rows.
                safe_schema = self.quote_identifier(schema_name)
                safe_table = self.quote_identifier(table_name)

                cursor.execute(
                    f"SELECT TOP 5 * FROM {safe_schema}.{safe_table}"
                )

                sample_data = cursor.fetchall()
                schema_info_context += "  Sample Data:\n"

                if not sample_data:
                    schema_info_context += "    No rows available.\n"
                else:
                    for row in sample_data:
                        schema_info_context += f"    {list(row)}\n"

            return schema_info_context

        except pyodbc.Error as e:
            raise RuntimeError(
                f"Error fetching schema details: {e}"
            ) from e

        finally:
            if cursor is not None:
                cursor.close()

    def execute_query(self, query, params=None):
        """
        Execute a SQL query.

        SELECT queries return a list of dictionaries.
        INSERT, UPDATE, and DELETE queries commit and return
        the number of affected rows.
        """

        if self.connection is None:
            raise RuntimeError("Database connection is not available.")

        cursor = None

        try:
            cursor = self.connection.cursor()

            # Execute the query, optionally with parameters.
            if params is None:
                cursor.execute(query)
            else:
                cursor.execute(query, params)

            # Queries that return columns, such as SELECT.
            if cursor.description is not None:
                column_names = [
                    column[0] for column in cursor.description
                ]

                rows = cursor.fetchall()

                results = [
                    dict(zip(column_names, row))
                    for row in rows
                ]

                return results

            # Queries that modify data.
            self.connection.commit()
            return cursor.rowcount

        except pyodbc.Error as e:
            self.connection.rollback()
            raise RuntimeError(
                f"Error executing query: {e}"
            ) from e

        finally:
            if cursor is not None:
                cursor.close()

    def close(self):
        """Close the connection when database operations are finished."""

        if self.connection is not None:
            self.connection.close()
            self.connection = None
            print("SQL Server connection closed.")


# Run this section only when database.py is executed directly.
# Importing DatabaseUtil from sql_analyst.py will not run this section.
if __name__ == "__main__":

    db_config = {
        "DRIVER": "ODBC Driver 18 for SQL Server",
        "SERVER": r"localhost\SQLEXPRESS",
        "DATABASE": "DATA_AGENT",
        "Trusted_Connection": "yes",
        "TrustServerCertificate": "yes"
    }

    db = DatabaseUtil(db_config)

    try:
        result = db.schema_details("dbo")

        with open("schema_info.txt", "w", encoding="utf-8") as file:
            file.write(result)

        print("Schema details saved to schema_info.txt")

    finally:
        db.close()
