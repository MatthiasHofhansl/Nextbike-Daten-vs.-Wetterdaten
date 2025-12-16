# Vor dem Start müssen Bibliotheken installiert werden.
# Folgendes muss hierzu vor dem Start in das Terminal eingegeben werden:
# pip install pandas

import sqlite3
import pandas as pd
import os

# Pfad zur Database (im übergeordneten Ordner, relativ zum Script)
# __file__ enthält den Pfad zur aktuell ausgeführten Datei. os.path.abspath wandelt den Pfad in einen absoluten Pfad um.
# os.path.dirname extrahiert nur das Verzeichnis (Also ohne den Dateinamen)
# Zeile darunter geht quasi ein Ordner hoch und dann wird hier nach der Database gesucht
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "..", "nextbike_data_old.db")

# Prüfen ob die Database existiert
if not os.path.exists(db_path):
    print(f"Fehler: Database nicht gefunden unter {db_path}")
    exit(1)

# Verbindung zur Database herstellen
conn = sqlite3.connect(db_path)

# Liste der zu exportierenden Tabellen
tables = ["bike_locations", "city_summaries", "stations"]

print("Starte Export der Tabellen...")

for table in tables:
    try:
        # Tabelle als DataFrame laden
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        
        # Als CSV speichern
        csv_filename = f"{table}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        
        # Informationen ausgeben
        rows = len(df)
        file_size = os.path.getsize(csv_filename) / (1024 * 1024)  # in MB
        print(f"✓ {table}: {rows} Zeilen exportiert → {csv_filename} ({file_size:.2f} MB)")
        
    except Exception as e:
        print(f"✗ Fehler beim Export von {table}: {e}")

# Verbindung schließen
conn.close()

print("\nExport abgeschlossen!")