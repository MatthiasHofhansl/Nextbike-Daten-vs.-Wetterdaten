import tkinter as tk
from tkinter import filedialog
import subprocess
import threading
import os

def start_interface():
    db_path = None

    def select_db():
        nonlocal db_path
        db_path = filedialog.askopenfilename(title="Datenbank auswählen", filetypes=[("SQLite files", "*.db")])
        if db_path:
            filename = os.path.basename(db_path)
            label.config(text=filename)
            btn_analyse.config(state='normal')

    def start_analysis():
        if db_path:
            def run_scripts():
                print("Speichere Nextbike-Daten in die Datenbank.")
                subprocess.run(["python", "get_nextbike_data.py", db_path])
                print("Speichere dazugehörige Wetterdaten in die Datenbank.")
                subprocess.run(["python", "weather_API.py", db_path])
                print("Nextbike-Daten und Wetterdaten erfolgreich abgespeichert.")
                print("Starte Analyse.")
                subprocess.run(["python", "analysis.py"])
                print("Analyse abgeschlossen.")
                root.quit()
            threading.Thread(target=run_scripts).start()

    root = tk.Tk()
    root.title("Verknüpfung von Nextbike-Daten mit Wetterdaten")

    # Set window size and center
    window_width = 500
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    btn_select = tk.Button(root, text="Datenbank auswählen", command=select_db)
    btn_select.pack(side='top', pady=20)

    label = tk.Label(root, text="", anchor='center')
    label.pack(pady=10)

    btn_analyse = tk.Button(root, text="Analyse starten", command=start_analysis, state='disabled')
    btn_analyse.pack(side='bottom', fill='x', pady=0)

    root.mainloop()