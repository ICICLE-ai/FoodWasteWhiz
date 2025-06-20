import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("feedback.db")

# Fetch all records into a DataFrame
df = pd.read_sql_query("SELECT * FROM feedback", conn)
print(df)

# Close the connection
conn.close()
