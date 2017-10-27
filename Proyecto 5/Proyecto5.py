# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:29:40 2017


@author: Cynthia Castillo Millán
Matrícula: A01374530

Red Neuronal BPN
Clasifica en dos clases, si la activación es sigmoidal
o predice el valor de la salida si la activación es lineal.
Implementa:
    * una sola neurona
    * Feedforward
    * Backpropagation para entranamiento 
    
Tiene tantas entradas como feautures tengan los ejemplos
y un layer de una salida de una sola neurona.

Tiene dos fases:
    1. Forward propagation: calcula Z, A y J
    2. Backpropagation: calcula dA, usando J
                        dZ usando A 
                        dW usando dZ y
                        dB usando dZ
                        
"""
# *****************************************************
import numpy as np
import random

def leerDatos(file):
    x1 = []
    x2 = []
    y = []
    data = open(file,"r")
    for line in data.readlines():
        x1.append(float(line.split(",")[0]))
        x2.append(float(line.split(",")[1]))
        y.append(float(line.split(",")[2]))
        
    xMatrix = np.asmatrix(np.column_stack((x1, x2)))
    yMatrix = np.asmatrix(np.column_stack(y))
    yMatrix = yMatrix.T
    
    return xMatrix, yMatrix


def sigmoid(z):      
    gZ = 1 / (1 + np.exp(-z))
    return gZ

"""
    ***En este proyecto, NO es necesario****
    * g'(z) = g(z)(1-g(z))
    -> g(z)=   1 / 1 + e^(-z)
    Los valores resultantes deben ser cercanos a 0
    Para z = 0, el valor debe ser 0.25
"""
def sigmoidGradiente(z):      
    gZ = 1 / (1 + np.exp(-z))
    dgZ = gZ*(1-gZ)
              
    return dgZ

def lineal(z):
    return z

"""
***En este proyecto, NO es necesario****
 El gradiente es la derivada.
 -> g(z) = z
    g'(z) = 1
"""
def linealGradiente(z):
    return 1

"""
    Inicializa aleatoriamente los pesos de una capa 
    con L_in entradas (unidades de capa anterior sin Bias)
    Los valores están en un rango de [-epsilon, epsilon]
    usar Epsilon = 0.12
"""
def randInicializaPesos(L_in):
    weights = np.zeros((L_in,1))
    epsilon = 0.12
    
    for i in range (0, L_in):
        weights[i] = random.uniform(-epsilon, epsilon)
    
    return weights

"""
    * nn_params: pesos
    * input_layer_size: Número de features
    * activacion: string que puede ser "sigmoidal" o lineal
    * Costo (con sigmoidal):
        J(theta) = 1/m sum( -yi log(h0(xi)) - (1-yi)log(1 - h0(xi)) )  --> de i a m
    * activacion:  un string que puede ser "sigmoidal" o "lineal"
    * h0 es la función sigmoidal
    * Costo (lineal):
        promedio de los errores al cuadrado (con el 2 para que se vaya en la derivada)
        
    * Esta función modifica el vector nn_params usando gradiente descendente utilizando
      el conjunto de ejemplos de entrenamiento ( X, y). 
    * Para calcular los gradientes utiliza backpropagation.    
"""
def bpnUnaNeurona(nn_params, input_layer_size, X, y, alpha, activacion):
    #X has the shape nxm (features x #examples)
    rows = input_layer_size #number of features
    cols = np.shape(X)[1] #number of examples
    J = 0
    W = randInicializaPesos(rows) # (rows x 1)
    b = 0
    dZ = 0
    dW = np.zeros((rows, 1)) #2 x 1
    dB = 0
    
    for a in range (0, 1000):
        J=0
        for i in range (0, cols):        
            #Forward Propagation
            Z = W.T * X[:, i] + b # (2x1).T (2,1) = 1
            if (activacion == "sigmoidal"):
                A = sigmoid(Z)
                #Cálculo del costo 
                first = -y[0,i] * np.log(A) #primer parte de la operacion
                second = (1-y[0,i]) * np.log(1-A) #segunda parte de la operacion
                result = first-second
                J += result
            else:
                A = lineal(Z)
                J += 0.5 * np.sum(np.square(np.subtract(y[0,i], A)))
        
            #Back propagtion
            """
            # *********************************************
            #Esto es exactamente igual que A-y, sin embargo,
            #se hace de manera directa
            #Esto es igual que dA
            dA = (-y[0, i]/A) + ((1-y[0, i])/(1-A))
            dZ = dA * sigmoidGradiente(Z)
            # *********************************************
            """
            dZ = A-y[0, i] #(1)
            #
            for f in range(0, rows):
                dW[f] = dW[f] + (X[f, i]*dZ) #(1)
            dB += dZ 
        J = J/cols
        dW = dW/cols
        dB = dB/cols
        #print(J)
        for f in range(0, rows):
                W[f] = W[f] - alpha*dW[f] #(1)
        b = b - alpha*dB
    return J, W , b

"""
    * nn_params: pesos
    * X: ejemplos que se desean clasificar, n features
    * y: prediccion de las clases para todos los  ejemplos de X
"""
def prediceRNYaEntrada(X, nn_params, b, activacion):

    yL = nn_params.T * X + b
    y = sigmoid(yL)
    
    if (activacion == "sigmoidal"):
        for i in range (0, y.shape[1]):
            if (y[0, i] >= 0.5): #se acepta
               y[0, i] = 1
            else: #no se acepta
               y[0, i] = 0
        return y
    
    else:
        return yL
    
def normalizacionDeCaracteristicas(X):
    rows = X.shape[0]
    cols = X.shape[1]
    sigma = []
    mu = []
    
    for j in range (0, cols):
        #standard deviation calculation
        sigma.append(np.std(X[0:rows,j]))
        #mean calculation
        mu.append(np.mean(X[0:rows,j]))
        #z-score per column
        for i in range (0, rows):
            X[i, j] = (X[i,j]-mu[j])/sigma[j]

    return X

def main():
    X, y = leerDatos("AND2.txt")
    X = normalizacionDeCaracteristicas(X)
    X = X.T
    y = y.T

    J, W, b= bpnUnaNeurona([], X.shape[0], X, y, 0.3, "lineal")
    X = np.asmatrix([8, 3]).T
    X = normalizacionDeCaracteristicas(X)
    print(X)
    a = prediceRNYaEntrada(X, W, b, "lineal")
    print(a)

    return 0

main()