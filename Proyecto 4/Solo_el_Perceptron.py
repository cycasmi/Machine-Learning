# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 02:08:44 2017

@author: Cynthia B. Castillo Millán
Matrícula: A01374530
"""
import  numpy as np
import random
import matplotlib.pyplot as plt

epsilon = 0.1

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

#Función que permite que los valores se encuentren entre 0 y 1
def sigmoidal(z):
    h0 = 1 / (1 + np.exp(-z))
    return h0

#Calcula el costo usando esa theta y calcula theta (con gradiente) con la theta inicial mandada
#Aqui, en lugar de enviar completo X, mando SOLO una fila, la que estoy utilizando
def funcionCostoPerceptron(theta, X, y):
    rows = np.shape(X)[0]
    J = []
    
    d = predicePerceptron(theta, X)
    errores = y - d
    for i in range (0, rows):
        J.append(errores[i, 0])

    return J

#Utiliza la función de costo para actualizar(entrenar) el perceptrón
#Recibe los casos, el resultado y un vector peso
def entrenaPerceptron(X, y, theta):
    #Valores inicializados de forma aleatoria
    #Nuevo ejemplo (xi1, xi2, di) donde d es la salida
    #Cálculo de salida con función escalón
    #Adaptación de pesos:
    #        wi(t+1) = wi(t) + alpha[d(t) - y(t)]xi(t) --> (0 <= i <= n)
    #    Alpha: es un factor de ganancia entre 0 y 1. Se ajusta a mano
    #    n:   es el número de pesos
    #Si error != (< epsilon) para todas las entradas, presentar un nuevo ejemplo
    cols = np.shape(X)[1]
    alpha = 1 #velocidad de aprendizaje
    
    J = []
    
    for a in range (0, 5000):
        error = funcionCostoPerceptron(theta, X, y) #obtiene el costo y actualiza los pesos
        J.append(error)
        for i in range (0, cols):
            if (abs(error[i]) > epsilon):
                theta[i] = theta[i] + alpha*error[i]*X[i, 0]
        if (np.sum(np.abs(error)) < epsilon):
            break
        
    return theta, J

#Dados los pesos obtenidos del perceptron y una entrada, se da un resultado
def predicePerceptron(theta, X): 
    rows = np.shape(X)[0]
    X = np.matrix(X)
    
    #Verifica si la matriz tiene una columna de unos o no, y si no, se los agrega
    if (X[0,0] != 1):
        ones = np.ones((rows,1))
        X = np.asmatrix(np.column_stack((ones, X)))
    
    p = sigmoidal(X*theta) #hace la prediccion
    
    for i in range (0, rows):
        if (p[i] >= 0.5): #se acepta
           p[i] = 1
        else: #no se acepta
           p[i] = 0
                 
    return p
    
    

#Calcula el costo usando esa theta y calcula theta (con gradiente) con la theta inicial mandada
def funcionCostoAdaline(theta,X, y):
    #El MEJOR vector de pesos es aquel que minimiza el error cuadrático medio
    #Es igual que el del perceptrón pero el error se calcula con Net y NO Y
    #       1/2L Sum(errk^2) --> de  k = 1 -> L
    # Donde: err = dk - sk
    # donde sk = Netk = XkW.T = Sum(wj xj k)  j=0 --> N
    rows = np.shape(X)[0]
    J = 0
    
    d= predicePerceptron(theta, X)
    err = np.square(y - d)
    J = (1/(2*rows))*np.sum(err)

    return J

#Utiliza la función de costo para actualizar(entrenar) el Adaline
#Recibe los casos, el resultado y un vector peso
def entrenaAdaline(X, y, theta):
    #wi = wi + alpha[dk - yk] xik
    #Donde:  k es el numero de ejemplo de la entrada
    #        i es el numero de entrada
    #        d es la salida deseada, en el ejemplo k
    #        y es la salida de k, con funcion lineal
    
    
    #Algoritmo
    #Inicializar pesos
    #Aplicar un vector de entrda Xk, en las entradas de adaline
    #obtener la salida lineal sk = Xk * W.T = Net
    #Calcular error ek = (dk-sk)
    #Actualizar pesos
    #repetir con todos los vectores de entrada
    #Si el error cuad. medio 1/2L... es un valor reducido aceptable, termina.
    #sino, se repiten los pasos.
    cols = np.shape(X)[1]
    alpha = 0.3 #velocidad de aprendizaje
    
    J = []
    for a in range (0, 5000):
        error = funcionCostoAdaline(theta, X, y) #obtiene el costo y actualiza los pesos
        J.append(error)
        for i in range (0, cols):
            theta[i] = theta[i] + alpha*error*X[0, i]
        if (error < epsilon):
            break
        
    return theta, J

#Dados los pesos obtenidos del perceptron y una entrada, se da un resultado
def prediceAdaline(theta, X):
    rows = np.shape(X)[0]
    X = np.matrix(X)
    
    #Verifica si la matriz tiene una columna de unos o no, y si no, se los agrega
    if (X[0,0] != 1):
        ones = np.ones((rows,1))
        X = np.asmatrix(np.column_stack((ones, X)))
    
    p = sigmoidal(X*theta) #hace la prediccion
    
    for i in range (0, rows):
        if (p[i] >= 0.5): #se acepta
           p[i] = 1
        else: #no se acepta
           p[i] = 0
                 
    return p

#AGREGAR ANÁLISIS Y COMENTARLO EN DOCUMENTACIÓN.
#HACE FALTA GUARDAR EL ERROR EN CADA PASO Y GRAFICARLO

X, y = leerDatos("OR.txt")
theta = np.asmatrix([1, 1, 1]).T                          


theta, J = entrenaPerceptron(X, y, theta)
print(theta)
#print(J)

p = predicePerceptron(theta, X)      
print (p) 


#theta, J = entrenaAdaline(X, y, theta)
#print(theta)
#print(J)

#p = prediceAdaline(theta, X)      
#print (p) 
                   