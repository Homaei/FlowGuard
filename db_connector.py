import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Handles connections to the PostgreSQL database and fetches water meter reading data.
    """
    
    def __init__(self):
        """Initialize database connection parameters from environment variables."""
        self.host = os.environ.get('DB_HOST', 'localhost')
        self.port = os.environ.get('DB_PORT', '5432')
        self.dbname = os.environ.get('DB_NAME', 'postgres')
        self.user = os.environ.get('DB_USER', 'postgres')
        self.password = os.environ.get('DB_PASSWORD', 'password')
        self.table_name = os.environ.get('DB_TABLE_NAME', 'uplinks')
        
    def get_connection(self):
        """Establish and return a database connection."""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def fetch_last_7_days_data(self):
        """
        Fetch data for the last 7 days from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing the fetched data.
        """
        conn = None
        try:
            conn = self.get_connection()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Format dates for query
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Fetching data from {start_str} to now...")
            
            # Construct query
            # Assuming columns: device_name, message_date, message_volume_registers
            query = f"""
                SELECT device_name, message_date, message_volume_registers
                FROM {self.table_name}
                WHERE message_date >= '{start_str}'
                ORDER BY message_date ASC
            """
            
            # Execute query and load into DataFrame
            df = pd.read_sql_query(query, conn)
            
            logger.info(f"Successfully fetched {len(df)} records.")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def fetch_data_for_device(self, device_name, days=7):
        """
        Fetch data for a specific device for the last N days.
        
        Args:
            device_name (str): The name/ID of the device.
            days (int): Number of days to look back.
            
        Returns:
            pd.DataFrame: DataFrame containing the fetched data.
        """
        conn = None
        try:
            conn = self.get_connection()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for query
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Fetching data for device '{device_name}' from {start_str}...")
            
            # Construct query
            query = f"""
                SELECT device_name, message_date, message_volume_registers
                FROM {self.table_name}
                WHERE device_name = '{device_name}'
                AND message_date >= '{start_str}'
                ORDER BY message_date ASC
            """
            
            # Execute query and load into DataFrame
            df = pd.read_sql_query(query, conn)
            
            logger.info(f"Successfully fetched {len(df)} records for device {device_name}.")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for device {device_name}: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    # Test connection (dry run)
    try:
        connector = DatabaseConnector()
        print("Database configuration:")
        print(f"Host: {connector.host}")
        print(f"Port: {connector.port}")
        print(f"DB Name: {connector.dbname}")
        print(f"User: {connector.user}")
        print(f"Table: {connector.table_name}")
        print("\nAttempting to connect (may fail if not on allowed network)...")
        # conn = connector.get_connection()
        # print("Connection successful!")
        # conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")
