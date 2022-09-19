import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

# Autor: Carlos Enrique Lucio Domínguez | A00828524
# Objetivo: Programación manual del algoritmo de aprendizaje máquina conocido como regresión lineal múltiple utilizando
# la técnica de Gradiente Descendiente.
# Problema: Encontrar un modelo de predicción para conocer el precio de un hogar a partir de características como número
# de recámaras, número de baños, tamaño del terreno total y de construcción, número de plantas/pisos, calificación de la
# vista y calificación de en qué condiciones se encuentra el hogar.
# Dataset Source: https://www.kaggle.com/datasets/shree1992/housedata?select=data.csv

df = pd.read_csv('training.csv') # Lectura del archivo de entrenamiento encontrado en el mismo folder que el archivo .py.

h   = lambda x1, x2, x3, x4, x5, x6, x7, theta: theta[0]+theta[1]*x1+theta[2]*x2+theta[3]*x3+theta[4]*x4+theta[5]*x5+theta[6]*x6+theta[7]*x7 # Función de hipótesis
j_i = lambda x1, x2, x3, x4, x5, x6, x7, y, theta: (h(x1, x2, x3, x4, x5, x6, x7, theta)-y)**2 # Funcion auxiliar de la función de costo

n_iter = 3000 # Número de iteraciones de aprendizaje
theta = [1,1,1,1,1,1,1,1] # Coeficientes del modelo
alpha = 0.000000001 # Tasa de aprendizaje

y = df['price']
x1 = df['bedrooms']
x2 = df['bathrooms']
x3 = df['sqft_living']
x4 = df['sqft_lot']
x5 = df['floors']
x6 = df['view']
x7 = df['condition']

n = len(y) # Cantidad de registros en el dataset de entrenamiento.

for idx in range(n_iter):

  delta = []
  deltaX1 = []
  deltaX2 = []
  deltaX3 = []
  deltaX4 = []
  deltaX5 = []
  deltaX6 = []
  deltaX7 = []

  for x1_i,x2_i,x3_i,x4_i,x5_i,x6_i,x7_i,y_i in zip(x1,x2,x3,x4,x5,x6,x7,y):

    yp_i = h(x1_i,x2_i,x3_i,x4_i,x5_i,x6_i,x7_i,theta)

    delta.append(yp_i-y_i)
    deltaX1.append((yp_i-y_i)*x1_i)
    deltaX2.append((yp_i-y_i)*x2_i)
    deltaX3.append((yp_i-y_i)*x3_i)
    deltaX4.append((yp_i-y_i)*x4_i)
    deltaX5.append((yp_i-y_i)*x5_i)
    deltaX6.append((yp_i-y_i)*x6_i)
    deltaX7.append((yp_i-y_i)*x7_i)

  # Actualización de los coeficientes de la función de hipótesis utilizando la técnica de Gradiente Descendiente.
  theta[0] = theta[0] - alpha/n*sum(delta)
  theta[1] = theta[1] - alpha/n*sum(deltaX1)
  theta[2] = theta[2] - alpha/n*sum(deltaX2)
  theta[3] = theta[3] - alpha/n*sum(deltaX3)
  theta[4] = theta[4] - alpha/n*sum(deltaX4)
  theta[5] = theta[5] - alpha/n*sum(deltaX5)
  theta[6] = theta[6] - alpha/n*sum(deltaX6)
  theta[7] = theta[7] - alpha/n*sum(deltaX7)

print("Modelo: " + str(round(theta[0], 4)) + " + " + str(round(theta[1], 4)) + "*bedrooms + "  + str(round(theta[2], 4)) + "*bathrooms + " + str(round(theta[3], 4)) + "*sqft_living + " + str(round(theta[4], 4)) + "*sqft_lot + " + str(round(theta[5], 4)) + "*floors + " + str(round(theta[6], 4)) + "*view + " + str(round(theta[7], 4)) + "*condition")
print()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Validación de error del modelo con los datos de entrenamiento y prueba.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dfT = pd.read_csv('testing.csv') # Lectura del archivo de testing encontrado en el mismo folder que el archivo .py.

y_test = dfT['price']
x1_test = dfT['bedrooms']
x2_test = dfT['bathrooms']
x3_test = dfT['sqft_living']
x4_test = dfT['sqft_lot']
x5_test = dfT['floors']
x6_test = dfT['view']
x7_test = dfT['condition']

# Calculo de Error de los datos de prueba
acum_delta_train = []
for x1_i,x2_i,x3_i,x4_i,x5_i,x6_i,x7_i,y_i in zip(x1,x2,x3,x4,x5,x6,x7,y):
  acum_delta_train.append(j_i(x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, y_i, theta))

sum_delta_train = sum(acum_delta_train)  
j_train = 1/(2*len(y))*sum_delta_train

# Impresion del Error con los datos de prueba
print('Error de datos de entrenamiento:', j_train)

# Calculo de Error de los datos de prueba
acum_delta_test = []
for x1_i,x2_i,x3_i,x4_i,x5_i,x6_i,x7_i,y_i in zip(x1_test,x2_test,x3_test,x4_test,x5_test,x6_test,x7_test,y_test):
  acum_delta_test.append(j_i(x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, y_i, theta))

sum_delta_test = sum(acum_delta_test)  
j_test = 1/(2*len(y_test))*sum_delta_test

# Impresion del Error con los datos de prueba
print('Error de datos de prueba:', j_test)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Realización de predicciones sobre un nuevo dataset (testing.csv) para la revisión del modelo.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Array de predicción de precios de los hogares.
y_p = []

for x1_i,x2_i,x3_i,x4_i,x5_i,x6_i,x7_i in zip(x1_test,x2_test,x3_test,x4_test,x5_test,x6_test,x7_test):
  y_p.append(h(x1_i,x2_i,x3_i,x4_i,x5_i,x6_i,x7_i,theta))

# Comparación entre valores reales y predicciones
comparison = pd.DataFrame(data={'Predicted Price': y_p, 'Real Price': y_test})
print('\nComparación de valores reales y predicciones:')
print(comparison.head())
print()

# Array de residuos.
residuals = y_p-y_test

# Gráfica de valores predichos vs valores reales
axes[0].scatter(y_test, y_p, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'black', lw=2)
axes[0].set_title('Valor Predicho vs Valor Real', fontsize = 10, fontweight = "bold")
axes[0].set_xlabel('Precio Real')
axes[0].set_ylabel('Precio Predicho')
axes[0].tick_params(labelsize = 6)

# Gráfica de residuos del modelo
axes[1].scatter(list(range(len(y_test))), residuals, edgecolors=(0, 0, 0), alpha = 0.4)
axes[1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[1].set_title('Residuos del Modelo', fontsize = 10, fontweight = "bold")
axes[1].set_xlabel('Id')
axes[1].set_ylabel('Residuo')
axes[1].tick_params(labelsize = 6)

plt.show()