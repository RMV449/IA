from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el escalador
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener datos del formulario
        pregnancies = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
        
        # Crear un array de características
        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        
        # Escalar los datos
        data_scaled = scaler.transform(data)
        
        # Hacer la predicción
        prediction = model.predict(data_scaled)
        
        # Mostrar el resultado
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        return render_template('index.html', result=result)
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
