import csv
import os
from pathlib import Path

import pyodbc
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

print("ENV file path:", BASE_DIR / ".env")
print("ENV file exists:", (BASE_DIR / ".env").is_file())
print("DB_SERVER loaded:", bool(os.getenv("DB_SERVER")))

CSV_DIR = BASE_DIR / "data"

required_env = [
    "DB_DRIVER", "DB_SERVER", "DB_DATABASE",
    "DB_TRUSTED_CONNECTION", "DB_TRUST_SERVER_CERTIFICATE"
]
missing = [key for key in required_env if not os.getenv(key)]
if missing:
    raise RuntimeError("Missing .env settings: " + ", ".join(missing))

connection_string = (
    f"DRIVER={{{os.environ['DB_DRIVER']}}};"
    f"SERVER={os.environ['DB_SERVER']};"
    f"DATABASE={os.environ['DB_DATABASE']};"
    f"Trusted_Connection={os.environ['DB_TRUSTED_CONNECTION']};"
    f"TrustServerCertificate={os.environ['DB_TRUST_SERVER_CERTIFICATE']};"
)

create_table_statements = [
    """
    IF OBJECT_ID('dbo.users', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.users (
            user_id INT IDENTITY(1,1) PRIMARY KEY,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            phone VARCHAR(50),
            city VARCHAR(100),
            province VARCHAR(50),
            user_type VARCHAR(20) NOT NULL,
            signup_date DATE,
            is_active BIT
        );
    END;
    """,
    """
    IF OBJECT_ID('dbo.vehicles', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.vehicles (
            vehicle_id INT IDENTITY(1,1) PRIMARY KEY,
            driver_id INT NOT NULL,
            make VARCHAR(50),
            model VARCHAR(50),
            year INT,
            license_plate VARCHAR(20) UNIQUE,
            color VARCHAR(30),
            is_active BIT,
            CONSTRAINT fk_vehicle_driver
                FOREIGN KEY (driver_id) REFERENCES dbo.users(user_id)
        );
    END;
    """,
    """
    IF OBJECT_ID('dbo.rides', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.rides (
            ride_id INT IDENTITY(1,1) PRIMARY KEY,
            rider_id INT NOT NULL,
            driver_id INT NOT NULL,
            CONSTRAINT fk_ride_rider
                FOREIGN KEY (rider_id) REFERENCES dbo.users(user_id),
            CONSTRAINT fk_ride_driver
                FOREIGN KEY (driver_id) REFERENCES dbo.users(user_id)
        );
    END;
    """
]


def load_csv(cursor, table_name, csv_file, columns):
    file_path = CSV_DIR / csv_file
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    col_names = ", ".join(f"[{column}]" for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = (
        f"INSERT INTO dbo.[{table_name}] ({col_names}) "
        f"VALUES ({placeholders})"
    )

    inserted_count = 0
    batch = []
    batch_size = 1000

    with file_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {file_path}")

        if header != columns:
            raise ValueError(
                f"Unexpected header in {csv_file}. Expected {columns}; found {header}"
            )

        for line_number, row in enumerate(reader, start=2):
            if not row or all(value == "" for value in row):
                continue
            if len(row) != len(columns):
                raise ValueError(
                    f"{csv_file}, line {line_number}: expected {len(columns)} "
                    f"values, found {len(row)}"
                )
            batch.append([None if value == "" else value for value in row])
            if len(batch) >= batch_size:
                cursor.executemany(insert_sql, batch)
                inserted_count += len(batch)
                batch.clear()

        if batch:
            cursor.executemany(insert_sql, batch)
            inserted_count += len(batch)

    print(f"Inserted {inserted_count} row(s) from {csv_file} into dbo.{table_name}.")


def main():
    for filename in ("users.csv", "vehicles.csv", "rides.csv"):
        file_path = CSV_DIR / filename
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing {file_path}. Put {filename} in the project's data folder."
            )

    conn = None
    cursor = None
    try:
        conn = pyodbc.connect(connection_string)
        conn.autocommit = False
        cursor = conn.cursor()
        print("Connected to SQL Server.")

        for statement in create_table_statements:
            cursor.execute(statement)
        print("Checked/created users, vehicles, and rides tables.")

        # Run against an empty test database. Re-running may violate UNIQUE
        # constraints on email and license_plate.
        load_csv(cursor, "users", "users.csv", [
            "first_name", "last_name", "email", "phone", "city",
            "province", "user_type", "signup_date", "is_active"
        ])
        load_csv(cursor, "vehicles", "vehicles.csv", [
            "driver_id", "make", "model", "year",
            "license_plate", "color", "is_active"
        ])
        load_csv(cursor, "rides", "rides.csv", ["rider_id", "driver_id"])

        conn.commit()
        print("SUCCESS: all CSV data was committed to SQL Server.")

        for table in ("users", "vehicles", "rides"):
            cursor.execute(f"SELECT COUNT(*) FROM dbo.[{table}]")
            print(f"dbo.{table}: {cursor.fetchone()[0]} total row(s)")

    except Exception:
        if conn is not None:
            conn.rollback()
        print("ERROR: loading failed. The current transaction was rolled back.")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
        print("SQL Server connection closed.")


if __name__ == "__main__":
    main()
