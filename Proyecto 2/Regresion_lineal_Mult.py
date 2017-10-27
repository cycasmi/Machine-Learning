# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:32:22 2017

@author: Cynthia B. Castillo Millán
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
    
    xMatrix = np.asmatrix(np.column_stack((x1, x2)))
    yMatrix = np.asmatrix(np.column_stack(y))
    yMatrix = yMatrix.T
    
    return xMatrix, yMatrix

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

    return X, mu, sigma

def gradienteDescendenteMultivariable(X, y, theta, alpha, iteraciones):
    rows = X.shape[0]
    cols = X.shape[1]
    J_Historia = []
    temp = theta 
    
    for i in range (0, iteraciones):
        for j in range (0, cols):
            h0 = (((X*theta)-y).T)*X[:,j]
            #derivative calculation
            temp[j] = theta[j] - (alpha/rows)*np.sum(h0)
        theta = temp
        
        #Recording error rate
        J_Historia.append(np.asscalar(calculaCosto(X, y, theta)))
        
        #Adjusting alpha
        if ( i > 0):
            if ((J_Historia[i] < J_Historia[i-1])>0.1):
                alpha = alpha*0.9

    return theta, J_Historia

def calculaCosto(X, y, theta):
    rows = X.shape[0]
    h0 = (X*theta)
    r = h0-y
    sums = 0
    
    #adding the errors, the cost function
    temp = np.square(r)
    for j in range (0,rows):
        sums = sums + temp[j]    
    return sums

def ecuacionNormal(X, y):
    theta = (X.T*X).I*X.T*y
    
    return theta

def predicePrecio(X, theta):
    precio =  X*theta
    return precio

def graficaError(J_Historial):
    plt.plot(J_Historial, 'r-')
    plt.show()
    return

dataFile = "data.txt"
X, y = leerDatos(dataFile)

theta = np.zeros((X.shape[1], 1))
alpha = 0.1
iteraciones = 100

#theta = ecuacionNormal(X, y)
#print(theta)

X, mu, sigma = normalizacionDeCaracteristicas(X)

a, J= gradienteDescendenteMultivariable(X, y, theta, alpha, iteraciones)
print(a)
graficaError(J)
