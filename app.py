import streamlit as st
import joblib
import numpy as np
import xgboost as xgb

# Cargar el modelo entrenado
modelo = xgb.XGBRegressor()
modelo.load_model("modelo_xgb.json")

# Función para calcular el BMI
def calcular_bmi(peso, altura):
    altura_m = altura / 100  # Convertir cm a metros
    return peso / (altura_m ** 2)

# Interfaz de usuario
st.title("Calculadora de Costo de Póliza 🚑")
st.write("Ingresa la información para estimar el costo anual de tu póliza de seguro.")

# Entradas del usuario
peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0)
altura = st.number_input("Altura (cm)", min_value=120, max_value=220, value=170)
fuma = st.selectbox("¿Fumas?", ["No", "Sí"])
hijos = st.number_input("Número de hijos", min_value=0, max_value=10, value=1)
edad = st.number_input("Edad", min_value=18, max_value=100, value=30)

# Convertir inputs en valores para el modelo
bmi = calcular_bmi(peso, altura)
bmi_smoker = bmi * (1 if fuma == "Sí" else 0)

# Crear un array con los valores de entrada
X_nuevo = np.array([[hijos, bmi_smoker, edad]])

# Predicción
if st.button("Calcular Costo"):
    costo_estimado = modelo.predict(X_nuevo)[0]
    costo_estimado = round(costo_estimado / 100) * 100  # Redondear al 100 más cercano
    st.success(f"El costo estimado de tu póliza anual es: **${costo_estimado}**")

