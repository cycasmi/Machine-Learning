# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:24:54 2017

@author: Cynthia Castillo Millán
Matrícula: A01374530

Red Neuronal BPN para detectar numeros a mano
El archivo digitos.txt, contiene 5000 ejemplos de dígitos
escritos a mano
*  Cada ejemplo de entrenamiento es una imagen en escala de
   grises (0 a 255) de 20X20 pixeles.
*  Cada imagen se desenrolló en una fila de 400 datos.
* Cada ejemplo está etiquetado del 1 al 10 (el 10 es 
  para el 0).
* El archivo lo deben separar en dos matrices:
    1. El vector X de 5000 X 400, conteniendo TODAS las 
       entradas. Se refiere a los pixeles de las imágenes.
       Se le debe agregar la columna de 1’s para procesarla.
    2. El vector y de 5000 X 1, conteniendo TODAS las 
       salidas. Se refieren al dígito que contiene la imagen.

* El archivo cero.txt, contiene los pesos de una RN ya 
  entrenada para reconocer dígitos, con la misma 
  arquitectura que la solicitada en esta actividad.
                        
"""

# *****************************************************
import numpy as np
import random

# ********** CAMBIOS!!!!!  ********************
# SE NECESITA PONER QUE SEA DE 1 A ELEMS-1, Y REGRESAR EL 1.0 INCIAL 
def leerDatos(file):
    X = [] #conjunto de todas las filas
    y = []
    data = open(file,"r")
    for line in data.readlines():
        elements = len(line.split(" "))
        if (elements > 1):
            X.append([])
            y.append(float(line.split(" ")[elements-1]))
        for i in range (0, elements-1):
            X[len(X)-1].append(float(line.split(" ")[i]))
    
    xMatrix = np.asmatrix(X)
    yMatrix = np.asmatrix(np.row_stack(y))
    
    return xMatrix, yMatrix

"""
    * g'(z) = g(z)(1-g(z))
    -> g(z)=   1 / 1 + e^(-z)
    Los valores resultantes deben ser cercanos a 0
    Para z = 0, el valor debe ser 0.25
"""
def sigmoidGradiente(z):      
    gZ = 1 / (1 + np.exp(-z))
    dgZ = gZ*(1-gZ)
              
    return dgZ

def sigmoid(z):      
    gZ = 1 / (1 + np.exp(-z))
    return gZ

"""
    Inicializa aleatoriamente los pesos de una capa 
    con L_in entradas (unidades de capa anterior sin Bias)
    Los valores están en un rango de [-epsilon, epsilon]
    usar Epsilon = 0.12
    L_out son cuantas salidas tiene (o cuantos vectores saldran
    a la siguiente capa)
"""
def randInicializaPesos(L_in, L_out):
    epsilon = 0.12
    weights = np.random.uniform(-epsilon, epsilon, (L_in, L_out))
    
    return weights

"""
    * input_layer_size: Unidades capa de entrada (Número de features + bias)
    * hidden_layer_size: número de parámetros de la capa oculta.
    * num_labels: (salidas) número de etiquetas en los ejemplos, en este caso es 10.    
"""
def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    #X has the shape (#examples x features)
    cols = input_layer_size #number of features
    rows = np.shape(X)[1] #number of examples
    J = 0 #Costo
    W1 = randInicializaPesos(cols, hidden_layer_size) # (features x capas intermedias) (400, 25)
    W2 = randInicializaPesos(hidden_layer_size, num_labels) # (capas intermedias x salidas) (25, 10)
    b1 = 0
    b2 = 0
    dZ = 0
    dW = np.zeros((1, cols)) #(1 x features) 
    dB = 0
    
    
    #Por cada ejemplo...
    for a in range (0, rows):
        Z1 = W1.T * X[a, :].T + b1 # (400x25).T * (1, 400).T = 25 x 1
        #print("Z1" , Z1.shape)
        A1 = sigmoid(Z1) #(25x1)
        #print("A1", A1.shape)
        Z2 = W2.T * A1 + b2 #(25.10).T * (25.1) = (10x1)
        #print("Z2" , Z2.shape)
        A2 = sigmoid(Z2)
        
        first = -y[a,0] * np.log(A1)
        print(first.shape)
        
        """
        if (activacion == "sigmoidal"):
            A = sigmoidGradiente(Z).T                 
            #Cálculo del costo 
            first = -y[0,i] * np.log(A) #primer parte de la operacion
            second = (1-y[0,i]) * np.log(1-A) #segunda parte de la operacion
            result = first-second
            J += result
        else:
            A = linealGradiente(Z)
            J += 0.5 * np.sum(np.square(np.subtract(y[0,i], A)))
    
        #Back propagtion
        dZ = y[0, i] - A #(1)
        for f in range(0, rows):
            dW[f] = dW[f] + (X[f, i]*dZ) #(1)
        dB += dZ 
        J = J/cols
        dW = dW/cols
        dB = dB/cols
        
        for f in range(0, rows):
                W[f] = W[f] - alpha*dW[f] #(1)
        b = b - alpha*dB
        """
    return J, W1

"""
    * nn_params: pesos
    * X: ejemplos que se desean clasificar, n features
    * y: prediccion de las clases para todos los  ejemplos de X
"""
def prediceRNYaEntrada(X, W1, b1, W2, b2):

    #y1 = sigmoid(W1.T *X +b1)
    #y2 = sigmoid(y1*W2+b2)
    #Result = max(y2)
    
    if (activacion == "sigmoidal"):
        for i in range (0, y.shape[1]):
            if (y[0, i] >= 0.5): #se acepta
               y[0, i] = 1
            else: #no se acepta
               y[0, i] = 0
  
    return y

def checkNNGradients():
    return 0

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
    #X = normalizacionDeCaracteristicas(X)
    
    input_layer_size = X.shape[1] #features, 400
    hidden_layer_size = 25
    num_labels = y.shape[0] #results, 10
    
    entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y)
    
    return 0

main()