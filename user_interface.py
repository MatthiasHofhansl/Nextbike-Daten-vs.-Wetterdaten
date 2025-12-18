import tkinter as tk
from tkinter import filedialog
import subprocess
import threading

def start_interface():
    def select_db():
        db_path = filedialog.askopenfilename(title="Select Database", filetypes=[("SQLite files", "*.db")])
        if db_path:
            def run_scripts():
                print("Starting get_nextbike_data.py...")
                subprocess.run(["python", "get_nextbike_data.py", db_path])
                print("Starting weather_API.py...")
                subprocess.run(["python", "weather_API.py", db_path])
                print("All scripts completed.")
                root.after(0, lambda: btn_analyse.config(state='normal'))
            threading.Thread(target=run_scripts).start()

    def start_analysis():
        subprocess.run(["python", "analysis.py"])
        root.destroy()

    root = tk.Tk()
    root.title("Verknüpfung von Nextbike-Daten mit Wetterdaten")

    # Set window size and center
    window_width = 400
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    btn_select = tk.Button(root, text="Select Database", command=select_db)
    btn_select.pack(pady=20)

    btn_analyse = tk.Button(root, text="Analyse starten", command=start_analysis, state='disabled')
    btn_analyse.pack(pady=20)

    root.mainloop()