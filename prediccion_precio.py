import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Descargar datos del ETF S&P 500 (SPY) desde 2018
data = yf.download('SPY', start='2018-01-01', end='2023-01-01')

# Mostrar los primeros 5 registros
print(data.head())

# Graficar el precio de cierre
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Precio de Cierre')
plt.title('Precio de Cierre de SPY (2018-2023)')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.show()

# Seleccionar solo la columna 'Close'
data = data[['Close']]

# Crear características basadas en los precios anteriores
data['Prev Close 1'] = data['Close'].shift(1)  # Cierre del día anterior
data['Prev Close 2'] = data['Close'].shift(2)  # Cierre de hace 2 días
data['Prev Close 3'] = data['Close'].shift(3)  # Cierre de hace 3 días

# Eliminar filas con valores nulos (debido al shift)
data = data.dropna()

# Mostrar las primeras filas para asegurarnos que todo esté bien
print(data.head())

from sklearn.model_selection import train_test_split

# Definir las características (X) y el objetivo (y)
X = data[['Prev Close 1', 'Prev Close 2', 'Prev Close 3']]
y = data['Close']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Ver las dimensiones de los conjuntos
print(f"Conjunto de entrenamiento: {X_train.shape}")
print(f"Conjunto de prueba: {X_test.shape}")


#ENTRENANDO EL MODELO
# Inicializar el modelo


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:5])


# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir los resultados
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Graficar las predicciones vs los valores reales
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Real', color='blue')
plt.plot(y_test.index, y_pred, label='Predicción', color='red')
plt.title('Predicción de Precios de SPY vs Real')
plt.xlabel('Fecha')
plt.ylabel('Precio ($)')
plt.legend()
plt.show()
