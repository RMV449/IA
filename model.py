import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    # Cargar datos
    training = pd.read_csv('data/Training.csv')
    X_train = training.drop(['Outcome'], axis=1)
    y_train = training['Outcome']
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Entrenar el modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Guardar el modelo y el escalador
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    train_model()
