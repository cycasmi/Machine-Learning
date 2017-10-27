# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:00:44 2017

@author: Cynthia Castillo Millán
Matrícula: A01374530
"""
import  numpy as np
import matplotlib.pyplot as plt

def leerDatos(file):
    x1 = []
    x2 = []
    y = []
    data = open(file,"r")
    for line in data.readlines():
        x1.append(float(line.split(",")[0]))
        x2.append(float(line.split(",")[1]))
        y.append(float(line.split(",")[2]))
    
    x = np.dot(x1,0)
    x = np.add(x,x+1)
    
    xMatrix = np.asmatrix(np.column_stack((x, x1, x2)))
    yMatrix = np.asmatrix(np.column_stack(y))
    yMatrix = yMatrix.T
    
    return xMatrix, yMatrix

def graficaDatos(X, y, theta):
    t = np.arange(1, 100, 1)
    rows = np.shape(X)[0]
    cols = np.shape(X)[1]
    
    #plot in blue dots the data and in a red line for the function
    plt.xlabel('Alumnos')
    plt.ylabel('Calificaciones 1')
    plt.title('Calificaciones 2')
    t = np.arange(0, 100, 1)

    plt.plot(X[:, 1:X.shape[1]], 'bo', sigmoidal(X*theta)*100, 'r-')
           
    plt.show()
    return 

#Función que permite que los valores se encuentren entre 0 y 1
def sigmoidal(z):
    h0 = 1 / (1 + np.exp(-z))
    return h0

#Calcula el costo usando esa theta y calcula theta (con gradiente) con la theta inicial mandada
def funcionCosto(theta, X, y):
    rows = np.shape(X)[0]
    cols = np.shape(X)[1]
    J = 0 #costo
    grad = np.zeros((cols,1)) #sera la actualizacion theta
    h0 = sigmoidal(X*theta) #hipotesis
   
    #Cálculo del costo 
    #print("y", y)
    print("h0",h0)
    first = np.asmatrix([float(a)*float(b) for a,b in zip(-y, np.log(h0))]).T #primer parte de la operacion
    second = np.asmatrix([float(a)*float(b) for a,b in zip(1-y, np.log(1-h0))]).T #segunda parte de la operacion
    result = first-second    
    J = np.sum(result)/rows #El calculo completo del costo
    
    h0_Y = h0-y
    for j in range (0, cols):
        summation = np.sum(np.asmatrix([float(a)*float(b) for a,b in zip(h0_Y, X[:,j])]).T) 
        grad[j] = summation/rows #Calculo del gradiente

    return J, grad

def aprende(theta, X, y, iteraciones):
    alpha = 0.001 #velocidad de aprendizaje
    
    for a in range (0, iteraciones):
        J, temp_theta = funcionCosto(theta, X, y) #obtiene el costo y la theta en cada iteracion
        theta = theta-(temp_theta * alpha) #actualiza la theta
        #print(J)
            
    return theta

#Se puede recibir X con o sin la columna de 1's
def predice(theta, X):
    rows = np.shape(X)[0]

    X = np.matrix(X)
    
    #Verifica si la matriz tiene una columna de unos o no, y si no, se los agrega
    if (X[0,0] != 1):
        ones = np.ones((rows,1))
        X = np.asmatrix(np.column_stack((ones, X)))
    
    h0 = sigmoidal(X*theta) #hace la prediccion
    if (h0 >= 0.5): #es  admitido
        return 1
    else: #reprueba
        return 0

dataFile = "data.txt"
X, y = leerDatos(dataFile)
theta = np.asmatrix(np.zeros(X.shape[1])).T
theta = aprende(theta, X, y, 100)
x = X[1,0:2]
x[0,0] = 45
x[0,1] = 85
print("\n\tSí fue admitido" if predice(theta, x) else "No fue admitido" )
graficaDatos(X,y,theta)
