import pyodbc
import sqlite3
import pandas as pd
import os

def export_sql_to_sqlite(sql_connection_string: str, sqlite_db_path: str):
    """
    Connects to a SQL Server database, fetches all base tables,
    and exports them to a local SQLite database.
    """
    print(f"Connecting to SQL Server...")
    try:
        sql_conn = pyodbc.connect(sql_connection_string)
    except Exception as e:
        print(f"Error connecting to SQL Server: {e}")
        raise

    # Remove existing sqlite file if it exists to start fresh
    if os.path.exists(sqlite_db_path):
        os.remove(sqlite_db_path)

    print(f"Creating/Connecting to SQLite database at {sqlite_db_path}...")
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    
    cursor = sql_conn.cursor()
    
    try:
        # Get list of tables
        # Note: This query works for SQL Server. 
        # For other DBs, the information schema query might differ.
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE='BASE TABLE'
        """)
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the source database.")
            return

        for (table_name,) in tables:
            print(f"Exporting table: {table_name}...")
            try:
                # Read from SQL Server
                df = pd.read_sql(f"SELECT * FROM [{table_name}]", sql_conn)
                
                # Write to SQLite
                # if_exists='replace' is safe here since we started with a fresh DB or want to overwrite
                df.to_sql(table_name, sqlite_conn, if_exists="replace", index=False)
                print(f"  -> Exported {len(df)} rows.")
            except Exception as e:
                print(f"  -> Failed to export {table_name}: {e}")

        print("All tables exported successfully!")
        
    finally:
        sql_conn.close()
        sqlite_conn.close()

if __name__ == "__main__":
    # Default credentials provided by user
    CONN_STR = (
        "Driver={SQL Server};"
        "Server=KYRILLOS\\SQLEXPRESS;"
        "Database=MyExcelDB;"
        "Trusted_Connection=yes;"
    )
    DB_PATH = "snapshot.db"
    
    export_sql_to_sqlite(CONN_STR, DB_PATH)
