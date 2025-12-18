# Vor dem Start müssen Bibliotheken installiert werden.
# Folgendes muss hierzu vor dem Start in das Terminal eingegeben werden:
# pip install pandas

import sqlite3
import pandas as pd
import os
import sys

# Pfad zur Database als Kommandozeilenargument
if len(sys.argv) < 2:
    print("Usage: python get_nextbike_data.py <database_path>")
    sys.exit(1)
db_path = sys.argv[1]

# Prüfen ob die Database existiert
if not os.path.exists(db_path):
    print(f"Fehler: Database nicht gefunden unter {db_path}")
    exit(1)

# Verbindung zur Database herstellen
conn = sqlite3.connect(db_path)

# Nur die Tabelle city_summaries wird exportiert
tables = ["city_summaries"]

for table in tables:
    try:
        # Tabelle als DataFrame laden
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

        # Timestamp parsen und date/hour extrahieren
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour

        # Nach date und hour gruppieren und Mittelwerte berechnen
        grouped = df.groupby(['date', 'hour']).agg({
            'total_bikes': 'mean',
            'available_bikes': 'mean',
            'booked_bikes': 'mean',
            'set_point_bikes': 'mean',
            'num_places': 'mean',
            'city_uid': 'first',
            'city_name': 'first',
            'country_name': 'first'
        }).reset_index()

        df = grouped

        # Runden der spezifischen Spalten und zu int konvertieren, um .0 zu vermeiden
        df['total_bikes'] = df['total_bikes'].round().astype(int)
        df['available_bikes'] = df['available_bikes'].round().astype(int)
        df['booked_bikes'] = df['booked_bikes'].round().astype(int)
        df['set_point_bikes'] = df['set_point_bikes'].round().astype(int)
        df['num_places'] = df['num_places'].round().astype(int)

        # Format hour as HH:MM:SS
        df['hour'] = df['hour'].apply(lambda h: f"{h:02d}:00:00")

        # Als CSV speichern
        csv_filename = f"{table}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')

    except Exception as e:
        print(f"✗ Fehler beim Export von {table}: {e}")

# Verbindung schließen
conn.close()