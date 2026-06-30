import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction Function
def predict_flower():
    try:
        values = [
            float(entry1.get()),
            float(entry2.get()),
            float(entry3.get()),
            float(entry4.get())
        ]

        prediction = model.predict([values])

        result.config(
            text="Predicted Flower: " + iris.target_names[prediction][0]
        )

    except:
        messagebox.showerror("Invalid Input", "Please enter valid numbers.")

# Main Window
root = tk.Tk()
root.title("AI Iris Flower Predictor")
root.geometry("500x450")
root.configure(bg="#EAF6F6")

# Heading
heading = tk.Label(
    root,
    text="🌸 AI Iris Flower Predictor 🌸",
    font=("Arial", 20, "bold"),
    bg="#EAF6F6",
    fg="darkblue"
)
heading.pack(pady=15)

# Labels and Entry Boxes
tk.Label(root, text="Sepal Length (cm)", font=("Arial", 12), bg="#EAF6F6").pack()
entry1 = tk.Entry(root, font=("Arial", 12), width=20)
entry1.pack(pady=5)

tk.Label(root, text="Sepal Width (cm)", font=("Arial", 12), bg="#EAF6F6").pack()
entry2 = tk.Entry(root, font=("Arial", 12), width=20)
entry2.pack(pady=5)

tk.Label(root, text="Petal Length (cm)", font=("Arial", 12), bg="#EAF6F6").pack()
entry3 = tk.Entry(root, font=("Arial", 12), width=20)
entry3.pack(pady=5)

tk.Label(root, text="Petal Width (cm)", font=("Arial", 12), bg="#EAF6F6").pack()
entry4 = tk.Entry(root, font=("Arial", 12), width=20)
entry4.pack(pady=5)

# Predict Button
predict_btn = tk.Button(
    root,
    text="Predict Flower",
    font=("Arial", 14, "bold"),
    bg="green",
    fg="white",
    padx=10,
    pady=5,
    command=predict_flower
)
predict_btn.pack(pady=20)

# Result
result = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"),
    bg="#EAF6F6",
    fg="blue"
)
result.pack()

root.mainloop()