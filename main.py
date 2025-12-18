import tkinter as tk
from tkinter import filedialog
import subprocess
import threading

def select_db():
    db_path = filedialog.askopenfilename(title="Select Database", filetypes=[("SQLite files", "*.db")])
    if db_path:
        def run_scripts():
            print("Starting get_nextbike_data.py...")
            subprocess.run(["python", "get_nextbike_data.py", db_path])
            print("Starting weather_API.py...")
            subprocess.run(["python", "weather_API.py", db_path])
            print("All scripts completed.")
        threading.Thread(target=run_scripts).start()

root = tk.Tk()
root.title("Nextbike Data Processor")
btn = tk.Button(root, text="Select Database", command=select_db)
btn.pack()
root.mainloop()