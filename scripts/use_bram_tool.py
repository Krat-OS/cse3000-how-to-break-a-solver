import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Button, filedialog, Label, Frame

def load_csv():
    """Load a CSV file and populate the button grid with features."""
    global df, feature_buttons
    file_path = filedialog.askopenfilename(
        title="Select a CSV File", 
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if not file_path:
        return

    df = pd.read_csv(file_path)
    feature_buttons = list(df.columns)
    display_features()

def display_features():
    """Display buttons in a grid layout for each feature in the loaded CSV."""
    for widget in button_frame.winfo_children():
        widget.destroy()  # Clear previous buttons

    for i, feature in enumerate(feature_buttons):
        button = Button(
            button_frame, text=feature, command=lambda f=feature: plot_feature(f),
            width=15, height=2, bg="#f0f0f0", fg="black"
        )
        button.grid(row=i // 6, column=i % 6, padx=5, pady=5)

def plot_feature(feature):
    """Plot a histogram for the selected feature."""
    try:
        values = df[feature].dropna()  # Exclude NaN values
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20, color='blue', edgecolor='black')
        plt.title(f"Histogram for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error plotting feature '{feature}': {e}")

# Initialize the GUI
root = Tk()
root.title("Feature Histogram Tool")
root.geometry("800x600")  # Adjusted for a better grid layout

# Add a label and a button to load CSV
load_label = Label(root, text="Select a CSV file to begin:", font=("Arial", 14))
load_label.pack(pady=10)

load_button = Button(root, text="Load CSV", command=load_csv, font=("Arial", 12), bg="#4CAF50", fg="white")
load_button.pack(pady=5)

# Add a frame for feature buttons
button_frame = Frame(root)
button_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Run the application
root.mainloop()
