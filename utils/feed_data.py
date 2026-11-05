import os
import csv
import pyodbc
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# SQL Server connection string building using updated .env variables
connection_string = (
    f"DRIVER={os.environ['DB_DRIVER']};"
    f"SERVER={os.environ['DB_SERVER']};"
    f"DATABASE={os.environ['DB_DATABASE']};"
    f"Trusted_Connection={os.environ['DB_TRUSTED_CONNECTION']};"
    f"TrustServerCertificate={os.environ['DB_TRUST_SERVER_CERTIFICATE']};"
)

CSV_DIR = "data"

# ==============================================================================
# DATABASE CONNECTION
# ==============================================================================

conn = pyodbc.connect(connection_string)
conn.autocommit = False

cursor = conn.cursor()

print("Connected to SQL Server")

# ==============================================================================
# CREATE TABLES & INDEXES (SQL Server 2025 Syntax)
# ==============================================================================

# 1. SQL Server uses 'dbo' as the default schema instead of 'public'.
# 2. 'IF NOT EXISTS' checks are adapted to T-SQL standards.
# 3. 'BOOLEAN' becomes 'BIT' (0 or 1).
# 4. 'PRIMARY KEY' integer columns use 'IDENTITY(1,1)' for auto-incrementing.

create_tables_sql = """
-- USERS
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

-- VEHICLES
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
        CONSTRAINT fk_vehicle_driver FOREIGN KEY (driver_id) REFERENCES dbo.users(user_id)
    );
END;

-- RIDES
IF OBJECT_ID('dbo.rides', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.rides (
        ride_id INT IDENTITY(1,1) PRIMARY KEY,
        rider_id INT NOT NULL,
        driver_id INT NOT NULL
        -- Add any missing fields shown lower in your original file here...
    );
END;

-- INDEXES
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_payments_ride_id' AND object_id = OBJECT_ID('dbo.payments'))
    CREATE INDEX idx_payments_ride_id ON dbo.payments(ride_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_payments_user_id' AND object_id = OBJECT_ID('dbo.payments'))
    CREATE INDEX idx_payments_user_id ON dbo.payments(user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ratings_ride_id' AND object_id = OBJECT_ID('dbo.ratings'))
    CREATE INDEX idx_ratings_ride_id ON dbo.ratings(ride_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ratings_driver_id' AND object_id = OBJECT_ID('dbo.ratings'))
    CREATE INDEX idx_ratings_driver_id ON dbo.ratings(driver_id);
"""

cursor.execute(create_tables_sql)
print("Tables created successfully")

# ==============================================================================
# DATA LOADING FUNCTION (Replaces PostgreSQL COPY)
# ==============================================================================

def load_csv(table_name, csv_file, columns):
    file_path = os.path.join(CSV_DIR, csv_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
        
    # Build standard T-SQL Parameterized Insert Query
    col_names = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = f"INSERT INTO dbo.{table_name} ({col_names}) VALUES ({placeholders})"
    
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip CSV header
        
        batch = []
        batch_size = 1000 # Keeps execution fast and memory usage optimized
        
        for row in reader:
            # Handle clean conversions (e.g. converting blank rows to NULL values)
            processed_row = [None if val == '' else val for val in row]
            batch.append(processed_row)
            
            if len(batch) >= batch_size:
                cursor.executemany(insert_sql, batch)
                batch = []
                
        # Insert any remaining records
        if batch:
            cursor.executemany(insert_sql, batch)
            
    print(f"Successfully loaded {csv_file} into dbo.{table_name}")

# ==============================================================================
# EXECUTION & CLOSING CONNECTIONS
# ==============================================================================

# Example executions (Ensure these match your actual column schemas)
# load_csv("users", "users.csv", ["first_name", "last_name", "email", "phone", "city", "province", "user_type", "signup_date", "is_active"])
# load_csv("vehicles", "vehicles.csv", ["driver_id", "make", "model", "year", "license_plate", "color", "is_active"])

conn.commit()
print("\nData loaded successfully!")
print("Transaction committed.")

# CLOSE CONNECTION
cursor.close()
conn.close()
print("SQL Server connection closed.")
