import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado
def load_model():
    # Aquí deberías cargar tu modelo previamente entrenado
    # Por simplicidad, se entrena un nuevo modelo en este ejemplo
    training = pd.read_csv("data/Training.csv")
    X_train = training.drop(['Outcome'], axis=1)
    y_train = training['Outcome']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

model, scaler = load_model()

def predict_diabetes():
    try:
        # Obtener valores del formulario
        age = float(age_entry.get())
        glucose = float(glucose_entry.get())
        blood_pressure = float(blood_pressure_entry.get())
        skin_thickness = float(skin_thickness_entry.get())
        insulin = float(insulin_entry.get())
        bmi = float(bmi_entry.get())
        diabetes_pedigree = float(diabetes_pedigree_entry.get())
        
        # Preparar los datos para la predicción
        input_data = pd.DataFrame([[age, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree]], 
                                  columns=['Age', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])
        
        input_data_scaled = scaler.transform(input_data)
        
        # Realizar la predicción
        prediction = model.predict(input_data_scaled)
        
        if prediction[0] == 1:
            result = "Positivo para Diabetes"
        else:
            result = "Negativo para Diabetes"
        
        # Mostrar el resultado
        messagebox.showinfo("Resultado", result)
    
    except ValueError:
        messagebox.showerror("Error", "Por favor ingrese valores válidos")

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Predicción de Diabetes")

# Etiquetas y campos de entrada
labels = ["Edad", "Glucosa", "Presión Arterial", "Grosor de Piel", "Insulina", "Índice de Masa Corporal (BMI)", "Función Pedigrí de Diabetes"]
entries = {}

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[label] = entry

age_entry = entries["Edad"]
glucose_entry = entries["Glucosa"]
blood_pressure_entry = entries["Presión Arterial"]
skin_thickness_entry = entries["Grosor de Piel"]
insulin_entry = entries["Insulina"]
bmi_entry = entries["Índice de Masa Corporal (BMI)"]
diabetes_pedigree_entry = entries["Función Pedigrí de Diabetes"]

# Botón para realizar la predicción
predict_button = tk.Button(root, text="Predecir", command=predict_diabetes)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

root.mainloop()
